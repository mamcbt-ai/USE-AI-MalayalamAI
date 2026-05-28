import os
import re
import numpy as np
import tempfile
import soundfile as sf
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "whisper-large-v3"
DEVICE = "groq-api"

print(f"ASR service ready: {MODEL_NAME} via Groq API")

# ---------------------------------------------------------------------------
# Tamil -> Malayalam Unicode corrector (kept as fallback)
# ---------------------------------------------------------------------------
_TAMIL_TO_ML = {
    '\u0B85': '\u0D05', '\u0B86': '\u0D06', '\u0B87': '\u0D07',
    '\u0B88': '\u0D08', '\u0B89': '\u0D09', '\u0B8A': '\u0D0A',
    '\u0B8E': '\u0D0E', '\u0B8F': '\u0D0F', '\u0B90': '\u0D10',
    '\u0B92': '\u0D12', '\u0B93': '\u0D13', '\u0B94': '\u0D14',
    '\u0B95': '\u0D15', '\u0B99': '\u0D19', '\u0B9A': '\u0D1A',
    '\u0B9C': '\u0D1C', '\u0B9E': '\u0D1E', '\u0B9F': '\u0D1F',
    '\u0BA3': '\u0D23', '\u0BA4': '\u0D24', '\u0BA8': '\u0D28',
    '\u0BA9': '\u0D29', '\u0BAA': '\u0D2A', '\u0BAE': '\u0D2E',
    '\u0BAF': '\u0D2F', '\u0BB0': '\u0D30', '\u0BB1': '\u0D31',
    '\u0BB2': '\u0D32', '\u0BB3': '\u0D33', '\u0BB4': '\u0D34',
    '\u0BB5': '\u0D35', '\u0BB6': '\u0D36', '\u0BB7': '\u0D37',
    '\u0BB8': '\u0D38', '\u0BB9': '\u0D39',
    '\u0BBE': '\u0D3E', '\u0BBF': '\u0D3F', '\u0BC0': '\u0D40',
    '\u0BC1': '\u0D41', '\u0BC2': '\u0D42', '\u0BC6': '\u0D46',
    '\u0BC7': '\u0D47', '\u0BC8': '\u0D48', '\u0BCA': '\u0D4A',
    '\u0BCB': '\u0D4B', '\u0BCC': '\u0D4C', '\u0BCD': '\u0D4D',
}
_TAMIL_RANGE = range(0x0B80, 0x0C00)


def _fix_ml_script(text: str) -> str:
    if not text:
        return text
    if any(ord(c) in _TAMIL_RANGE for c in text):
        return ''.join(_TAMIL_TO_ML.get(c, c) for c in text)
    return text


def _postprocess_script(text: str, source_lang: str) -> str:
    if source_lang == "ml":
        return _fix_ml_script(text)
    return text


def cleanup_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\.{2,}", ".", text)
    text = text.strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text


def _load_audio(audio_input):
    if isinstance(audio_input, np.ndarray):
        return audio_input
    audio, _ = sf.read(audio_input, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio


def _to_wav_bytes(audio_input) -> bytes:
    print(f"[ASR] _to_wav_bytes: type={type(audio_input)}, is_ndarray={isinstance(audio_input, np.ndarray)}")
    """Convert audio_input (ndarray or file path) to WAV bytes for Groq API."""
    if isinstance(audio_input, np.ndarray):
        audio = audio_input
    else:
        audio, sr = sf.read(audio_input, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        wav_bytes = f.read()
    os.unlink(tmp_path)
    return wav_bytes


_LANG_NAMES = {
    "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada",   "hi": "Hindi",
}

# Languages where we force language=None on local model (Groq handles natively)
# For Groq, we always pass the language explicitly
_GROQ_LANG_MAP = {
    "ml": "ml", "ta": "ta", "te": "te", "kn": "kn", "hi": "hi",
}


def _call_groq_translate(wav_bytes: bytes) -> str:
    """Pass 1: speech -> English translation."""
    translation = client.audio.translations.create(
        file=("audio.wav", wav_bytes),
        model=MODEL_NAME,
        response_format="text",
    )
    return cleanup_text(translation.text.strip() if hasattr(translation, "text") else str(translation).strip())


def _call_groq_transcribe(wav_bytes: bytes, source_lang: str) -> str:
    """Pass 2: speech -> native script transcription."""
    lang = _GROQ_LANG_MAP.get(source_lang, source_lang)
    transcription = client.audio.transcriptions.create(
        file=("audio.wav", wav_bytes),
        model=MODEL_NAME,
        language=lang,
        response_format="text",
    )
    raw = (transcription.text if hasattr(transcription, "text") else str(transcription)).strip()
    return cleanup_text(_postprocess_script(raw, source_lang))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_audio(audio_input, source_lang: str = "ml"):
    try:
        wav_bytes = _to_wav_bytes(audio_input)
        lang_name = _LANG_NAMES.get(source_lang, source_lang.upper())
        print(f"[ASR] Groq API call: lang={source_lang}, audio={len(wav_bytes)} bytes")

        print(f"[ASR] wav_bytes size={len(wav_bytes)}")
        print(f"[ASR] calling Groq translate...")
        english_text = _call_groq_translate(wav_bytes)
        print(f"[ASR] translate result: {repr(english_text)}")
        print(f"[ASR] calling Groq transcribe lang={source_lang}...")
        native_text  = _call_groq_transcribe(wav_bytes, source_lang)
        print(f"[ASR] transcribe result: {repr(native_text)}")

        print(f"English        : {english_text}")
        print(f"{lang_name:<14} : {native_text}")

        return {
            "status": "success",
            "text": english_text,
            "malayalam_text": native_text,
            "raw_text": english_text,
            "language": source_lang,
            "source_lang": source_lang,
            "segments": [],
            "device": DEVICE,
            "model": MODEL_NAME,
        }
    except Exception as e:
        print(f"ASR Error: {e}")
        return {
            "status": "failed", "error": str(e),
            "text": "", "malayalam_text": "", "segments": [],
        }


def transcribe_audio_stream(audio_input, source_lang: str = "ml"):
    """
    Groq is fast enough that we call both passes then yield results.
    The SSE stream fires all events within ~3 seconds total.
    """
    try:
        wav_bytes = _to_wav_bytes(audio_input)
        print(f"[ASR] Groq stream: lang={source_lang}, audio={len(wav_bytes)} bytes")

        # Pass 1: English
        print(f"[ASR] wav_bytes size={len(wav_bytes)}")
        print(f"[ASR] calling Groq translate...")
        english_text = _call_groq_translate(wav_bytes)
        print(f"[ASR] translate result: {repr(english_text)}")
        if english_text:
            yield {"type": "english_segment", "text": english_text,
                   "accumulated": english_text}

        # Pass 2: Native script
        native_text = _call_groq_transcribe(wav_bytes, source_lang)
        if native_text:
            yield {"type": "malayalam_segment", "text": native_text,
                   "accumulated": native_text}

        yield {
            "type": "complete",
            "english_text": english_text,
            "malayalam_text": native_text,
            "language": source_lang,
            "source_lang": source_lang,
        }
    except Exception as e:
        print(f"ASR Stream Error: {e}")
        yield {"type": "error", "error": str(e)}