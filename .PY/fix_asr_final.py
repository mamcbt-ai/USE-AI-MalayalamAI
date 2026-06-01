path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend\app\services\asr_service.py'

content = r"""import os
import re
import numpy as np
import tempfile
import soundfile as sf
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL = "llama-3.3-70b-versatile"
DEVICE = "groq-api"

print(f"ASR ready: {WHISPER_MODEL} + {LLM_MODEL} via Groq API")

LANG_NAMES = {
    "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada",   "hi": "Hindi",
}

HALLUCINATIONS = [
    "thank you for watching", "thanks for watching", "subscribe",
    "music", "[music]", "subtitles", "captions",
    "hello and welcome", "welcome to my channel", "translated by",
    "english is a language", "language of the language",
    "i'm here with a story", "hello everyone",
]

def cleanup_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()

def _is_hallucination(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    from collections import Counter
    counts = Counter(text.replace(" ", ""))
    if counts:
        ratio = counts.most_common(1)[0][1] / max(len(text.replace(" ", "")), 1)
        if ratio > 0.6:
            return True
    lower = text.lower()
    if any(lower.startswith(p) or (p in lower and len(text) < 80) for p in HALLUCINATIONS):
        return True
    words = text.split()
    if len(words) >= 4:
        if len(set(w.lower() for w in words)) / len(words) < 0.5:
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

def _load_audio(audio_input) -> np.ndarray:
    if isinstance(audio_input, np.ndarray):
        return audio_input
    audio, _ = sf.read(audio_input, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio

def _whisper_transcribe(wav_bytes: bytes, lang: str) -> str:
    """Transcribe in native language using Whisper"""
    try:
        result = groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            language=lang,
            response_format="text",
        )
        text = result.text if hasattr(result, "text") else str(result)
        return cleanup_text(text.strip())
    except Exception as e:
        print(f"[ASR] Transcribe error: {e}")
        return ""

def _llm_translate_to_english(native_text: str, lang: str) -> str:
    """LLM: native language text -> English"""
    if not native_text:
        return ""
    lang_name = LANG_NAMES.get(lang, lang)
    try:
        resp = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": f"You are an expert {lang_name} to English translator. Translate the given {lang_name} text to natural English. Output ONLY the English translation, no explanation."},
                {"role": "user", "content": native_text},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        result = resp.choices[0].message.content.strip()
        print(f"[ASR] LLM English: {result[:80]}")
        return cleanup_text(result)
    except Exception as e:
        print(f"[ASR] LLM translate error: {e}")
        return ""

def _llm_to_native_unicode(native_text: str, lang: str) -> str:
    """LLM: ensure text is proper Unicode script (fix transliteration if any)"""
    if not native_text:
        return ""
    lang_name = LANG_NAMES.get(lang, lang)
    # Check if already in native script (non-ASCII dominant)
    ascii_ratio = sum(1 for c in native_text if ord(c) < 128) / max(len(native_text), 1)
    if ascii_ratio < 0.3:
        return native_text  # Already mostly native Unicode
    # Re-translate if it came back in Roman/English
    try:
        prompts = {
            "ml": "Convert this text to natural Malayalam Unicode script. Output ONLY Malayalam characters.",
            "ta": "Convert this text to natural Tamil Unicode script. Output ONLY Tamil characters.",
            "te": "Convert this text to natural Telugu Unicode script. Output ONLY Telugu characters.",
            "kn": "Convert this text to natural Kannada Unicode script. Output ONLY Kannada characters.",
            "hi": "Convert this text to natural Hindi Devanagari script. Output ONLY Hindi characters.",
        }
        resp = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": prompts.get(lang, prompts["ml"])},
                {"role": "user", "content": native_text},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        return cleanup_text(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"[ASR] Unicode fix error: {e}")
        return native_text

def transcribe_audio(audio_input, style: str = "standard", source_lang: str = "ml") -> dict:
    try:
        wav_bytes = _to_wav_bytes(_load_audio(audio_input))
        print(f"[ASR] lang={source_lang} bytes={len(wav_bytes)}")

        # Step 1: Transcribe in native language (more accurate than translation)
        native_raw = _whisper_transcribe(wav_bytes, source_lang)
        print(f"[ASR] Whisper native: {native_raw}")

        if _is_hallucination(native_raw):
            native_raw = ""

        # Step 2: LLM translate native -> English
        english_text = _llm_translate_to_english(native_raw, source_lang) if native_raw else ""
        print(f"[ASR] English: {english_text}")

        # Step 3: Ensure native is proper Unicode
        native_text = _llm_to_native_unicode(native_raw, source_lang) if native_raw else ""
        print(f"[ASR] Native : {native_text}")

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
        wav_bytes = _to_wav_bytes(_load_audio(audio_input))
        print(f"[ASR] stream lang={source_lang} bytes={len(wav_bytes)}")

        native_raw = _whisper_transcribe(wav_bytes, source_lang)
        if _is_hallucination(native_raw):
            native_raw = ""
        print(f"[ASR] Whisper native: {native_raw}")

        english_text = _llm_translate_to_english(native_raw, source_lang) if native_raw else ""
        print(f"[ASR] English: {english_text}")

        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}

        native_text = _llm_to_native_unicode(native_raw, source_lang) if native_raw else ""
        print(f"[ASR] Native : {native_text}")

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
"""

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Done')
