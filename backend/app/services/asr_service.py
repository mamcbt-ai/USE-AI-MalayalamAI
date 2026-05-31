import os
import re
import numpy as np
import tempfile
import soundfile as sf
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
WHISPER_MODEL = "whisper-large-v3"
DEVICE = "groq-api"

print(f"ASR ready: {WHISPER_MODEL} via Groq API")

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
    if not text or len(text) < 3:
        return False
    from collections import Counter
    counts = Counter(text.replace(" ", ""))
    if counts:
        ratio = counts.most_common(1)[0][1] / max(len(text.replace(" ", "")), 1)
        if ratio > 0.6:
            return True
    # Whisper hallucination phrases
    bad = [
        "thank you for watching", "thanks for watching", "subscribe",
        "music", "[music]", "subtitles", "captions",
        "thank you", "hello and welcome", "welcome to my channel",
        "hello, welcome", "translated by", "translation by",
        "english is a language", "language of the language",
    ]
    lower = text.lower()
    if any(lower.startswith(p) or (p in lower and len(text) < 60) for p in bad):
        return True
    # Detect word repetition (e.g. "kar do kar do kar do")
    words = text.split()
    if len(words) >= 4:
        unique = len(set(w.lower() for w in words))
        if unique / len(words) < 0.5:
            return True
    return False

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

LANG_NAMES = {
    "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "hi": "Hindi",
}

def _groq_transcribe(wav_bytes: bytes, lang: str) -> str:
    """Native script via Groq Whisper large-v3 transcription."""
    try:
        result = groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            language=lang,
            response_format="text",

        )
        text = result.text if hasattr(result, "text") else str(result)
        result = cleanup_text(text.strip())
        # Filter out if Groq echoed back the prompt
        prompt_echo = f"This is {LANG_NAMES.get(lang, lang)} speech"
        if result.lower().startswith(prompt_echo.lower()[:20]):
            return ""
        return result
    except Exception as e:
        print(f"[ASR] Transcribe error: {e}")
        return ""

def _groq_translate(wav_bytes: bytes, lang: str) -> str:
    """English translation via Groq Whisper large-v3."""
    try:
        result = groq_client.audio.translations.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            response_format="text",
            prompt=f"Translate this {LANG_NAMES.get(lang, lang)} speech to English, including colloquial and slang expressions.",
        )
        text = result.text if hasattr(result, "text") else str(result)
        return cleanup_text(text.strip())
    except Exception as e:
        print(f"[ASR] Translate error: {e}")
        return ""

def transcribe_audio(audio_input, style: str = "standard", source_lang: str = "ml") -> dict:
    try:
        wav_bytes = _to_wav_bytes(audio_input)
        print(f"[ASR] lang={source_lang} bytes={len(wav_bytes)}")

        english_text = _groq_translate(wav_bytes, source_lang)
        print(f"[ASR] English  : {english_text}")

        native_text = _groq_transcribe(wav_bytes, source_lang)
        if _is_hallucination(native_text):
            native_text = ""
        print(f"[ASR] Native   : {native_text}")

        return {
            "status": "success",
            "text": english_text,
            "malayalam_text": native_text,
            "raw_text": english_text,
            "language": source_lang,
            "segments": [],
            "device": DEVICE,
            "model": WHISPER_MODEL,
        }
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return {"status": "failed", "error": str(e), "text": "", "malayalam_text": "", "segments": []}

def transcribe_audio_stream(audio_input, style: str = "standard", source_lang: str = "ml"):
    try:
        wav_bytes = _to_wav_bytes(audio_input)
        print(f"[ASR] stream lang={source_lang} bytes={len(wav_bytes)}")

        english_text = _groq_translate(wav_bytes, source_lang)
        if _is_hallucination(english_text):
            english_text = ""
        print(f"[ASR] English  : {english_text}")

        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}

        native_text = _groq_transcribe(wav_bytes, source_lang)
        if _is_hallucination(native_text):
            native_text = ""
        print(f"[ASR] Native   : {native_text}")

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