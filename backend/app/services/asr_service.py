"""
asr_service.py — Multilingual ASR via Groq Whisper large-v3-turbo
Supports: Malayalam (ml), Tamil (ta), Telugu (te), Kannada (kn), Hindi (hi)
Response contract: english_text, native_text, source_lang, source_language_name

Architecture:
  - Raw audio bytes sent directly to Groq (no PyAV conversion needed)
  - Groq handles WebM, MP4, OGG, WAV natively
  - verbose_json for better text extraction and language detection
  - Hallucination filtering only (no script gating)
"""
import os
import re
from collections import Counter
from typing import Any, Dict, Generator

from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQAPIKEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")  # Best multilingual quality
DEVICE        = "groq-api"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

groq_client = Groq(api_key=GROQ_API_KEY)

LANGUAGE_NAMES: Dict[str, str] = {
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "hi": "Hindi",
}

# Unicode blocks per language — for logging/validation
SCRIPT_PATTERNS: Dict[str, str] = {
    "ml": r"[ഀ-ൿ]",
    "ta": r"[஀-௿]",
    "te": r"[ఀ-౿]",
    "kn": r"[ಀ-೿]",
    "hi": r"[ऀ-ॿ]",
}

HALLUCINATION_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "subtitles",
    "captions",
    "[music]",
    "[ music ]",
    "romantic music",
    "hello and welcome",
    "welcome to my channel",
    "translated by",
    "translation by",
]

print(f"ASR: Using Groq API ({WHISPER_MODEL})")


# ── Style prompts ─────────────────────────────────────────────────────────────
_STYLE_PROMPTS: Dict[str, str] = {
    "standard":      "",  # No prompt — let Groq be natural
    "formal":        "Formal speech.",
    "casual":        "Casual conversational speech.",
    "business":      "Business meeting speech.",
    "academic":      "Academic lecture speech.",
    "news":          "News broadcast speech.",
    "literary":      "Literary narration.",
    "simple":        "Simple everyday speech.",
    "humorous":      "Humorous speech.",
    "emotional":     "Emotional speech.",
    "bullet":        "List-style speech.",
}


def _get_style_prompt(style: str, lang_name: str) -> str:
    base = _STYLE_PROMPTS.get(style, "")
    if base:
        return f"This is {lang_name} {base}"
    return f"This is {lang_name} speech."


# ── Text helpers ──────────────────────────────────────────────────────────────
def cleanup_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.!?])\1+", r"\1", text)
    return text.strip()


def is_hallucination(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return True
    lower = text.lower().strip()
    if any(p in lower for p in HALLUCINATION_PHRASES):
        return True
    compact = lower.replace(" ", "")
    if compact and len(compact) > 5:
        counts = Counter(compact)
        ratio = counts.most_common(1)[0][1] / max(len(compact), 1)
        if ratio > 0.65:
            return True
    words = lower.split()
    if len(words) >= 4 and len(set(words)) / len(words) < 0.35:
        return True
    return False


def has_native_script(text: str, lang: str) -> bool:
    pattern = SCRIPT_PATTERNS.get(lang)
    if not pattern:
        return True
    return bool(re.search(pattern, text))


def _extract_from_verbose(result: Any) -> str:
    """Extract text from verbose_json or text response."""
    if result is None:
        return ""
    if hasattr(result, "text"):
        return cleanup_text(result.text or "")
    return cleanup_text(str(result))


# ── Groq API calls ────────────────────────────────────────────────────────────
def groq_transcribe_native(raw_bytes: bytes, source_lang: str,
                            filename: str = "recording.webm", style: str = "standard") -> str:
    """Transcribe audio in native script. Sends raw bytes directly to Groq."""
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    prompt = _get_style_prompt(style, lang_name)
    print(f"Groq native: lang={source_lang}, bytes={len(raw_bytes)}, file={filename}")
    # Force native script output in prompt
    script_prompts = {
        "ml": "ഈ ശബ്ദം മലയാളത്തിൽ ലിപ്യന്തരണം ചെയ്യുക.",  # Transcribe in Malayalam script
        "ta": "இந்த ஒலியை தமிழில் எழுதுக.",
        "te": "ఈ శబ్దాన్ని తెలుగులో రాయండి.",
        "kn": "ಈ ಧ್ವನಿಯನ್ನು ಕನ್ನಡದಲ್ಲಿ ಬರೆಯಿರಿ.",
        "hi": "इस आवाज़ को हिंदी में लिखें।",
    }
    native_prompt = script_prompts.get(source_lang, prompt)
    try:
        result = groq_client.audio.transcriptions.create(
            file=(filename, raw_bytes),
            model=WHISPER_MODEL,
            language=source_lang,
            response_format="verbose_json",
            temperature=0.0,
            prompt=native_prompt,
        )
        text = _extract_from_verbose(result)
        detected = getattr(result, "language", source_lang)
        print(f"Groq native result (detected={detected}): '{text[:120] if text else '(empty)'}'")
        has_script = has_native_script(text, source_lang)
        if not has_script:
            print(f"  -> Warning: no {lang_name} script chars in output")
        return text
    except Exception as e:
        print(f"Groq native ERROR: {type(e).__name__}: {e}")
        return ""


def translate_to_english_gpt(native_text: str, source_lang: str, style: str = "standard") -> str:
    """
    Translate native text to English using GPT-4o-mini.
    whisper-large-v3-turbo does NOT support the translate task,
    so we use GPT instead for better quality translation.
    """
    if not native_text or not native_text.strip():
        return ""

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("translate_to_english_gpt: no OPENAI_API_KEY — returning native text as-is")
        return native_text

    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    from app.services.translation_service import STYLE_INSTRUCTIONS
    style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["standard"])

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are an expert translator from {lang_name} to English. "
                        f"{style_instruction} "
                        "Return only the English translation — no explanations."
                    ),
                },
                {"role": "user", "content": native_text},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        result = response.choices[0].message.content.strip()
        print(f"GPT-4o-mini translation ({style}): '{result[:100]}'")
        return result
    except Exception as e:
        print(f"GPT translate error: {type(e).__name__}: {e}")
        if "insufficient_quota" in str(e) or "429" in str(e):
            print("GPT quota exceeded — returning empty (not copying native text)")
        return ""  # Never copy native text into English field


# ── Public API ────────────────────────────────────────────────────────────────
def transcribe_audio(audio_input: Any, filename: str = "recording.webm",
                     style: str = "standard", source_lang: str = "ml") -> Dict[str, Any]:
    """
    Transcribe audio and return stable language-neutral dict.
    audio_input: bytes (raw file) or file path string
    """
    if source_lang not in LANGUAGE_NAMES:
        return {
            "status": "failed", "error": f"Unsupported language: {source_lang}",
            "english_text": "", "native_text": "",
            "source_lang": source_lang, "source_language_name": source_lang,
            "style": style, "segments": [], "device": DEVICE, "model": WHISPER_MODEL,
        }

    lang_name = LANGUAGE_NAMES[source_lang]

    try:
        # Get raw bytes
        if isinstance(audio_input, (bytes, bytearray)):
            raw_bytes = bytes(audio_input)
        else:
            with open(str(audio_input), "rb") as f:
                raw_bytes = f.read()
            import os as _os
            ext = _os.path.splitext(str(audio_input))[1] or ".webm"
            filename = f"audio{ext}"

        print(f"ASR: lang={source_lang}, style={style}, bytes={len(raw_bytes)}, file={filename}")

        if len(raw_bytes) < 1000:
            return {
                "status": "too_short",
                "error": "Recording too short. Please speak for at least 3 seconds.",
                "english_text": "", "native_text": "",
                "source_lang": source_lang, "source_language_name": lang_name,
                "style": style, "segments": [], "device": DEVICE, "model": WHISPER_MODEL,
            }

        # Pass 1: Native script transcription (turbo model)
        native_text = groq_transcribe_native(raw_bytes, source_lang, filename, style)

        # Filter hallucinations
        if is_hallucination(native_text):
            print(f"Native hallucination rejected: {native_text[:60]}")
            native_text = ""

        # Pass 2: English translation via GPT-4o-mini (turbo doesn't support translate)
        english_text = translate_to_english_gpt(native_text, source_lang, style) if native_text else ""

        translation_status = "success" if english_text else "unavailable"
        print(f"Final — English ({translation_status}): '{english_text[:80] if english_text else '(empty)'}' | Native: '{native_text[:80] if native_text else '(empty)'}'")

        return {
            "status": "success" if native_text else "no_speech",
            "english_text": english_text,
            "native_text":  native_text,
            "source_lang":  source_lang,
            "source_language_name": lang_name,
            "translation_status": translation_status,
            "style":    style,
            "segments": [],
            "device":   DEVICE,
            "model":    WHISPER_MODEL,
        }
    except Exception as e:
        print(f"ASR Error: {e}")
        return {
            "status": "failed", "error": str(e),
            "english_text": "", "native_text": "",
            "source_lang": source_lang, "source_language_name": lang_name,
            "style": style, "segments": [], "device": DEVICE, "model": WHISPER_MODEL,
        }


def transcribe_audio_stream(audio_input: Any, filename: str = "recording.webm",
                             style: str = "standard", source_lang: str = "ml") -> Generator[Dict, None, None]:
    """Streaming variant — yields SSE-compatible dicts."""
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        if isinstance(audio_input, (bytes, bytearray)):
            raw_bytes = bytes(audio_input)
        else:
            with open(str(audio_input), "rb") as f:
                raw_bytes = f.read()

        yield {"type": "status", "message": f"Processing {lang_name} audio..."}

        native_text  = groq_transcribe_native(raw_bytes, source_lang, filename, style)
        if is_hallucination(native_text):
            native_text = ""
        english_text = translate_to_english_gpt(native_text, source_lang, style) if native_text else ""

        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}
        if native_text:
            yield {"type": "native_segment", "text": native_text, "accumulated": native_text}

        yield {
            "type": "complete",
            "english_text": english_text,
            "native_text":  native_text,
            "source_lang":  source_lang,
            "source_language_name": lang_name,
            "style": style,
        }
    except Exception as e:
        print(f"ASR stream error: {e}")
        yield {"type": "error", "error": str(e)}
