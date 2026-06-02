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
    """
    Return True if text contains at least some characters from the expected script.
    Relaxed: only reject if there are ZERO native chars (not based on ratio).
    """
    if not text:
        return False
    pattern = SCRIPT_PATTERNS.get(lang)
    if not pattern:
        return True  # unknown lang — accept
    script_chars = len(re.findall(pattern, text))
    return script_chars > 0  # accept as long as ANY native char is present


# ── Audio helpers ─────────────────────────────────────────────────────────────
def to_wav_bytes(audio_input: Any) -> bytes:
    """
    Convert audio input → 16 kHz mono WAV bytes for Groq API.
    Handles: numpy array, file path (WebM/MP4/WAV/etc), raw bytes.
    Uses PyAV to decode WebM/Opus from browser MediaRecorder.
    """
    if isinstance(audio_input, np.ndarray):
        audio = audio_input
    else:
        # Resolve to file path
        if isinstance(audio_input, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio_input)
                file_path = tmp.name
            owns_file = True
        else:
            file_path = str(audio_input)
            owns_file = False

        audio = None

        # Try PyAV first (handles WebM/Opus from browser)
        try:
            import av
            import io as _io
            with open(file_path, "rb") as f:
                raw = f.read()
            container = av.open(_io.BytesIO(raw))
            samples = []
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                samples.append(arr.astype(np.float32))
            if samples:
                audio = np.concatenate(samples)
                if audio.max() > 1.5:
                    audio = audio / 32768.0
                print(f"PyAV decode OK: {audio.shape}, max={audio.max():.3f}")
        except Exception as e:
            print(f"PyAV decode failed: {e}")

        # Fallback: soundfile (handles WAV, FLAC, OGG)
        if audio is None:
            try:
                audio, _ = sf.read(file_path, dtype="float32")
                print(f"soundfile decode OK: {audio.shape}")
            except Exception as e:
                print(f"soundfile decode failed: {e}")

        if owns_file and os.path.exists(file_path):
            os.unlink(file_path)

        if audio is None:
            raise ValueError("Could not decode audio file — unsupported format")

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Normalize peak
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
    Returns: english_text, native_text, native_text_raw, source_lang, source_language_name
    """
    # Validate language
    if source_lang not in LANGUAGE_NAMES:
        return {
            "status": "failed", "error": f"Unsupported language: {source_lang}",
            "english_text": "", "native_text": "", "native_text_raw": "",
            "source_lang": source_lang, "source_language_name": source_lang,
            "style": style, "segments": [], "device": DEVICE, "model": WHISPER_MODEL,
        }

    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        wav_bytes = to_wav_bytes(audio_input)
        print(f"ASR: lang={source_lang}, style={style}, bytes={len(wav_bytes)}")

        # Minimum length: ~1 second at 16kHz 16-bit = ~32KB
        if len(wav_bytes) < 32000:
            return {
                "status": "too_short",
                "error": "Recording too short. Please speak for at least 3 seconds.",
                "english_text": "", "native_text": "", "native_text_raw": "",
                "source_lang": source_lang, "source_language_name": lang_name,
                "style": style, "segments": [], "device": DEVICE, "model": WHISPER_MODEL,
            }

        native_text_raw = groq_transcribe_native(wav_bytes, source_lang)
        english_text    = groq_translate_english(wav_bytes, source_lang)
        native_text     = native_text_raw  # already filtered in groq_transcribe_native

        print(f"ASR English     : {english_text[:80] if english_text else '(empty)'}")
        print(f"ASR Native (raw): {native_text_raw[:80] if native_text_raw else '(empty)'}")

        return {
            "status": "success",
            "english_text":    english_text,
            "native_text":     native_text,
            "native_text_raw": native_text_raw,
            "source_lang":     source_lang,
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
