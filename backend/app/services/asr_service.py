"""
asr_service.py — Language-neutral ASR + Translation
Supports: Malayalam (ml), Tamil (ta), Telugu (te), Kannada (kn), Hindi (hi)

Uses Groq Whisper API if GROQ_API_KEY is set (fast, cloud, large-v3).
Falls back to local faster-whisper small on CPU if no key is set.
"""
import os
import re
import tempfile
import numpy as np
from collections import Counter

# ── Language registry ─────────────────────────────────────────────────────────
LANGUAGE_NAMES = {
    "ml": "Malayalam",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "hi": "Hindi",
}

# ── Backend selection: Groq API (preferred) or local Whisper (fallback) ───────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
USE_GROQ = bool(GROQ_API_KEY)

if USE_GROQ:
    from groq import Groq
    _groq = Groq(api_key=GROQ_API_KEY)
    WHISPER_MODEL = "whisper-large-v3"
    DEVICE = "groq-cloud"
    print(f"ASR: Using Groq API ({WHISPER_MODEL})")
else:
    import torch
    from faster_whisper import WhisperModel
    WHISPER_MODEL = "small"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _local_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type="int8")
    print(f"ASR: Using local Whisper ({WHISPER_MODEL}/{DEVICE}) — set GROQ_API_KEY for better accuracy")


# ── Text helpers ──────────────────────────────────────────────────────────────
def cleanup_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\.{2,}", ".", text)
    text = text.strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text


def _is_hallucination(text: str) -> bool:
    """Return True if text looks like Whisper hallucination (repeated chars, stock phrases)."""
    if not text or len(text.strip()) < 3:
        return False
    cleaned = text.replace(" ", "")
    if cleaned:
        counts = Counter(cleaned)
        ratio = counts.most_common(1)[0][1] / len(cleaned)
        if ratio > 0.6:
            print(f"Hallucination (repeated chars {ratio:.0%}): {text[:50]}")
            return True
    lower = text.lower()
    stock = ["thank you for watching", "thanks for watching", "subscribe",
             "♪", "[music]", "[ music ]", "subtitles", "captions",
             "english translation", "formal english", "casual english",
             "simple english", "humorous english", "english bullet"]
    if any(p in lower for p in stock) and len(text) < 60:
        print(f"Hallucination (stock phrase): {text[:50]}")
        return True
    return False


# ── Audio loading ─────────────────────────────────────────────────────────────
def _load_audio_numpy(audio_input) -> np.ndarray:
    """Load audio to float32 numpy array at 16 kHz mono."""
    if isinstance(audio_input, np.ndarray):
        return audio_input
    import soundfile as sf
    audio, _ = sf.read(audio_input, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio


def _numpy_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert numpy float32 array → WAV bytes (for Groq API)."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        wav_bytes = f.read()
    os.unlink(tmp_path)
    return wav_bytes


# ── Groq API transcription ────────────────────────────────────────────────────
def _groq_translate_english(wav_bytes: bytes, source_lang: str, style: str) -> str:
    """Use Groq Whisper to translate source language → English."""
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    style_hints = {
        "formal": "formal", "casual": "casual", "business": "business",
        "academic": "academic", "news": "news broadcast", "literary": "literary",
        "simple": "simple plain", "humorous": "humorous", "emotional": "emotional",
    }
    style_desc = style_hints.get(style, "accurate")
    try:
        result = _groq.audio.translations.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            response_format="text",
            prompt=f"Translate this {lang_name} speech to {style_desc} English.",
        )
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        return cleanup_text(text)
    except Exception as e:
        print(f"Groq translate error: {e}")
        return ""


def _groq_transcribe_native(wav_bytes: bytes, source_lang: str) -> str:
    """Use Groq Whisper to transcribe in source language (native script)."""
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        result = _groq.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            language=source_lang,
            response_format="text",
            prompt=f"This is {lang_name} speech including colloquial expressions.",
        )
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        return cleanup_text(text)
    except Exception as e:
        print(f"Groq transcribe error: {e}")
        return ""


# ── Local Whisper transcription ───────────────────────────────────────────────
def _local_translate_english(audio: np.ndarray, source_lang: str, style: str) -> str:
    style_prompts = {
        "formal": "Formal English translation.", "casual": "Casual English translation.",
        "business": "Business English translation.", "academic": "Academic English translation.",
        "news": "News style English translation.", "simple": "Simple English translation.",
        "humorous": "Humorous English translation.", "emotional": "Emotional English translation.",
    }
    prompt = style_prompts.get(style, "English translation.")
    try:
        segs, _ = _local_model.transcribe(
            audio, task="translate", language=source_lang,
            initial_prompt=prompt, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            beam_size=5, temperature=0,
        )
        parts = [s.text.strip() for s in segs if s.text.strip()]
        text = cleanup_text(" ".join(parts))
        return "" if _is_hallucination(text) else text
    except Exception as e:
        print(f"Local translate error: {e}")
        return ""


def _local_transcribe_native(audio: np.ndarray, source_lang: str) -> str:
    try:
        segs, _ = _local_model.transcribe(
            audio, task="transcribe", language=source_lang,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
            beam_size=5, temperature=0,
        )
        parts = [s.text.strip() for s in segs if s.text.strip()]
        text = cleanup_text(" ".join(parts))
        return "" if _is_hallucination(text) else text
    except Exception as e:
        print(f"Local transcribe error: {e}")
        return ""


# ── Public API ────────────────────────────────────────────────────────────────
def transcribe_audio(audio_input, style: str = "standard", source_lang: str = "ml") -> dict:
    """
    Transcribe audio and return language-neutral response dict.
    Returns: { status, english_text, native_text, source_lang, source_language_name, ... }
    """
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        audio = _load_audio_numpy(audio_input)
        print(f"Transcribing: shape={audio.shape}, max={audio.max():.3f}, lang={source_lang}, style={style}")

        if USE_GROQ:
            wav_bytes = _numpy_to_wav_bytes(audio)
            english_text = _groq_translate_english(wav_bytes, source_lang, style)
            native_text  = _groq_transcribe_native(wav_bytes, source_lang)
        else:
            english_text = _local_translate_english(audio, source_lang, style)
            native_text  = _local_transcribe_native(audio, source_lang)

        if _is_hallucination(native_text):
            native_text = ""

        print(f"English : {english_text}")
        print(f"Native  : {native_text}")

        return {
            "status": "success",
            "english_text": english_text,
            "native_text": native_text,
            "source_lang": source_lang,
            "source_language_name": lang_name,
            "segments": [],
            "device": DEVICE,
            "model": WHISPER_MODEL,
        }
    except Exception as e:
        print(f"ASR Error: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "english_text": "",
            "native_text": "",
            "source_lang": source_lang,
            "source_language_name": lang_name,
            "segments": [],
            "device": DEVICE,
            "model": WHISPER_MODEL,
        }


def transcribe_audio_stream(audio_input, style: str = "standard", source_lang: str = "ml"):
    """
    Streaming version — yields SSE-compatible dicts.
    Event types: english_segment, native_segment, complete, error
    """
    lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    try:
        audio = _load_audio_numpy(audio_input)

        if USE_GROQ:
            wav_bytes = _numpy_to_wav_bytes(audio)
            english_text = _groq_translate_english(wav_bytes, source_lang, style)
            native_text  = _groq_transcribe_native(wav_bytes, source_lang)
        else:
            english_text = _local_translate_english(audio, source_lang, style)
            native_text  = _local_transcribe_native(audio, source_lang)

        if _is_hallucination(native_text):
            native_text = ""

        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}

        if native_text:
            yield {"type": "native_segment", "text": native_text, "accumulated": native_text}

        yield {
            "type": "complete",
            "english_text": english_text,
            "native_text": native_text,
            "source_lang": source_lang,
            "source_language_name": lang_name,
        }
    except Exception as e:
        print(f"ASR stream error: {e}")
        yield {"type": "error", "message": str(e)}
