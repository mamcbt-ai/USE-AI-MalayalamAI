import os
import re
import numpy as np
import tempfile
import soundfile as sf
from groq import Groq
import anthropic

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))

GROQ_MODEL   = "whisper-large-v3"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
DEVICE = "groq+claude"

print(f"ASR ready: {GROQ_MODEL} (translate) + {CLAUDE_MODEL} (native script)")

# ---------------------------------------------------------------------------
# Language config
# ---------------------------------------------------------------------------
_LANG_NAMES = {
    "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada",   "hi": "Hindi",
}

_CLAUDE_PROMPTS = {
    "ml": "Translate the following English text to Malayalam. Output ONLY Malayalam Unicode script (ഇങ്ങനെ), no English, no romanization, no explanation:\n\n",
    "ta": "Translate the following English text to Tamil. Output ONLY Tamil Unicode script (இப்படி), no English, no romanization, no explanation:\n\n",
    "te": "Translate the following English text to Telugu. Output ONLY Telugu Unicode script (ఇలా), no English, no romanization, no explanation:\n\n",
    "kn": "Translate the following English text to Kannada. Output ONLY Kannada Unicode script (ಹೀಗೆ), no English, no romanization, no explanation:\n\n",
    "hi": "Translate the following English text to Hindi. Output ONLY Hindi Devanagari script (इस तरह), no English, no romanization, no explanation:\n\n",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cleanup_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


def _to_wav_bytes(audio_input) -> bytes:
    if isinstance(audio_input, np.ndarray):
        audio = audio_input
    else:
        audio, _ = sf.read(audio_input, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        wav_bytes = f.read()
    os.unlink(tmp_path)
    return wav_bytes


def _groq_translate(wav_bytes: bytes) -> str:
    """Pass 1: speech -> English via Groq Whisper translate."""
    result = groq_client.audio.translations.create(
        file=("audio.wav", wav_bytes),
        model=GROQ_MODEL,
        response_format="text",
    )
    text = result.text if hasattr(result, "text") else str(result)
    return cleanup_text(text.strip())


def _claude_to_native(english_text: str, source_lang: str) -> str:
    """Pass 2: English -> native Unicode script via Claude Haiku."""
    if not english_text:
        return ""
    prompt = _CLAUDE_PROMPTS.get(source_lang, _CLAUDE_PROMPTS["ml"]) + english_text
    message = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return cleanup_text(message.content[0].text.strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def transcribe_audio(audio_input, source_lang: str = "ml"):
    try:
        wav_bytes = _to_wav_bytes(audio_input)
        lang_name = _LANG_NAMES.get(source_lang, source_lang.upper())
        print(f"[ASR] lang={source_lang} bytes={len(wav_bytes)}")

        english_text = _groq_translate(wav_bytes)
        print(f"[ASR] English  : {english_text}")

        native_text = _claude_to_native(english_text, source_lang)
        print(f"[ASR] {lang_name:<10}: {native_text}")

        return {
            "status": "success",
            "text": english_text,
            "malayalam_text": native_text,
            "raw_text": english_text,
            "language": source_lang,
            "source_lang": source_lang,
            "segments": [],
            "device": DEVICE,
            "model": GROQ_MODEL,
        }
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return {"status": "failed", "error": str(e), "text": "", "malayalam_text": "", "segments": []}


def transcribe_audio_stream(audio_input, source_lang: str = "ml"):
    try:
        wav_bytes = _to_wav_bytes(audio_input)
        lang_name = _LANG_NAMES.get(source_lang, source_lang.upper())
        print(f"[ASR] stream lang={source_lang} bytes={len(wav_bytes)}")

        # Pass 1: Groq → English
        english_text = _groq_translate(wav_bytes)
        print(f"[ASR] English  : {english_text}")
        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}

        # Pass 2: Claude → native Unicode
        native_text = _claude_to_native(english_text, source_lang)
        print(f"[ASR] {lang_name:<10}: {native_text}")
        if native_text:
            yield {"type": "malayalam_segment", "text": native_text, "accumulated": native_text}

        yield {
            "type": "complete",
            "english_text": english_text,
            "malayalam_text": native_text,
            "language": source_lang,
            "source_lang": source_lang,
        }
    except Exception as e:
        print(f"[ASR] Stream Error: {e}")
        yield {"type": "error", "error": str(e)}