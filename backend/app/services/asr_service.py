"""
asr_service.py — Multilingual ASR via Groq Whisper large-v3
Supports: Malayalam (ml), Tamil (ta), Telugu (te), Kannada (kn), Hindi (hi)
Response contract: english_text, native_text, source_lang, source_language_name
"""
import os
import re
import tempfile
from collections import Counter
from typing import Any, Dict, Generator

import numpy as np
import soundfile as sf
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQAPIKEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")
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

# Unicode blocks for each language — used to validate native script output
SCRIPT_PATTERNS: Dict[str, str] = {
    "ml": r"[ഀ-ൿ]",   # Malayalam
    "ta": r"[஀-௿]",   # Tamil
    "te": r"[ఀ-౿]",   # Telugu
    "kn": r"[ಀ-೿]",   # Kannada
    "hi": r"[ऀ-ॿ]",   # Devanagari (Hindi)
}

HALLUCINATION_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "subtitles",
    "captions",
    "music",
    "romantic music",
    "hello and welcome",
    "welcome to my channel",
    "translated by",
    "translation by",
]

print(f"ASR: Using Groq API ({WHISPER_MODEL})")


# ── Text helpers ──────────────────────────────────────────────────────────────
def cleanup_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.!?])\1+", r"\1", text)
    return text.strip()


def is_hallucination(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return True
    lower = text.lower().strip()
    if any(p in lower for p in HALLUCINATION_PHRASES):
        return True
    compact = lower.replace(" ", "")
    if compact:
        counts = Counter(compact)
        ratio = counts.most_common(1)[0][1] / max(len(compact), 1)
        if ratio > 0.60:
            return True
    # Repeated word pattern (e.g. "the the the the")
    words = lower.split()
    if len(words) >= 4 and len(set(words)) / len(words) < 0.4:
        return True
    return False


def has_expected_script(text: str, lang: str) -> bool:
    """Return True only if text contains characters from the expected Unicode script."""
    if not text:
        return False
    pattern = SCRIPT_PATTERNS.get(lang)
    if not pattern:
        return True  # Hindi and others — fall back to accepting
    script_chars = len(re.findall(pattern, text))
    latin_chars  = len(re.findall(r"[A-Za-z]", text))
    # Require at least one native char AND more native than Latin
    return script_chars > 0 and script_chars >= latin_chars


# ── Audio helpers ─────────────────────────────────────────────────────────────
def to_wav_bytes(audio_input: Any) -> bytes:
    """Convert file path or numpy array → 16 kHz mono WAV bytes."""
    if isinstance(audio_input, np.ndarray):
        audio = audio_input
    elif isinstance(audio_input, (bytes, bytearray)):
        # Raw bytes — write to temp file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_input)
            tmp_path = tmp.name
        try:
            audio, _ = sf.read(tmp_path, dtype="float32")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        audio, _ = sf.read(audio_input, dtype="float32")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Normalize peak to avoid clipping distortion
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _extract_text(result: Any) -> str:
    if result is None:
        return ""
    if hasattr(result, "text"):
        return cleanup_text(result.text or "")
    return cleanup_text(str(result))


# ── Groq API calls ────────────────────────────────────────────────────────────
def groq_transcribe_native(wav_bytes: bytes, source_lang: str) -> str:
    """Transcribe audio in native script using Groq Whisper."""
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        result = groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            language=source_lang,
            response_format="text",
            prompt=f"This is {lang_name} speech including colloquial and slang expressions.",
        )
        text = _extract_text(result)
        if is_hallucination(text):
            print(f"Native hallucination rejected: {text[:60]}")
            return ""
        if not has_expected_script(text, source_lang):
            print(f"Native script mismatch ({source_lang}), text rejected: {text[:60]}")
            return ""
        return text
    except Exception as e:
        print(f"Groq native transcription error: {e}")
        return ""


def groq_translate_english(wav_bytes: bytes, source_lang: str) -> str:
    """Translate audio to English using Groq Whisper."""
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        result = groq_client.audio.translations.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            response_format="text",
            prompt=f"Translate this {lang_name} speech accurately into natural English.",
        )
        text = _extract_text(result)
        if is_hallucination(text):
            print(f"English hallucination rejected: {text[:60]}")
            return ""
        return text
    except Exception as e:
        print(f"Groq English translation error: {e}")
        return ""


# ── Public API ────────────────────────────────────────────────────────────────
def transcribe_audio(audio_input: Any, style: str = "standard", source_lang: str = "ml") -> Dict[str, Any]:
    """
    Transcribe audio and return stable language-neutral dict.
    Always returns: english_text, native_text, source_lang, source_language_name
    """
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        wav_bytes = to_wav_bytes(audio_input)
        print(f"ASR: lang={source_lang}, style={style}, bytes={len(wav_bytes)}")

        native_text  = groq_transcribe_native(wav_bytes, source_lang)
        english_text = groq_translate_english(wav_bytes, source_lang)

        print(f"ASR English : {english_text[:80] if english_text else '(empty)'}")
        print(f"ASR Native  : {native_text[:80] if native_text else '(empty)'}")

        return {
            "status": "success",
            "english_text": english_text,
            "native_text":  native_text,
            "source_lang":  source_lang,
            "source_language_name": lang_name,
            "style":    style,
            "segments": [],
            "device":   DEVICE,
            "model":    WHISPER_MODEL,
        }
    except Exception as e:
        print(f"ASR Error: {e}")
        return {
            "status": "failed",
            "error":  str(e),
            "english_text": "",
            "native_text":  "",
            "source_lang":  source_lang,
            "source_language_name": lang_name,
            "style":    style,
            "segments": [],
            "device":   DEVICE,
            "model":    WHISPER_MODEL,
        }


def transcribe_audio_stream(
    audio_input: Any,
    style: str = "standard",
    source_lang: str = "ml",
) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming variant — yields SSE-compatible dicts.
    Event types: status, english_segment, native_segment, complete, error
    """
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        wav_bytes = to_wav_bytes(audio_input)

        yield {"type": "status", "message": f"Processing {lang_name} audio..."}

        english_text = groq_translate_english(wav_bytes, source_lang)
        native_text  = groq_transcribe_native(wav_bytes, source_lang)

        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}
        if native_text:
            yield {"type": "native_segment",  "text": native_text,  "accumulated": native_text}

        yield {
            "type":   "complete",
            "english_text": english_text,
            "native_text":  native_text,
            "source_lang":  source_lang,
            "source_language_name": lang_name,
            "style":  style,
        }
    except Exception as e:
        print(f"ASR stream error: {e}")
        yield {"type": "error", "error": str(e)}
