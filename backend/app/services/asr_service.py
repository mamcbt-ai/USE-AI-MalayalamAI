import os
import re
import numpy as np
import tempfile
import soundfile as sf
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL     = "llama-3.1-8b-instant"
DEVICE        = "groq-api"

print(f"ASR ready: {WHISPER_MODEL} (translate) + {LLM_MODEL} (native script)")

_LANG_NAMES = {
    "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada",   "hi": "Hindi",
}

_NATIVE_PROMPTS = {
    "ml": "Translate the following English text to Malayalam Unicode script. Output ONLY Malayalam characters (like: നമസ്കാരം). No English, no explanation.",
    "ta": "Translate the following English text to Tamil Unicode script. Output ONLY Tamil characters (like: வணக்கம்). No English, no explanation.",
    "te": "Translate the following English text to Telugu Unicode script. Output ONLY Telugu characters (like: నమస్కారం). No English, no explanation.",
    "kn": "Translate the following English text to Kannada Unicode script. Output ONLY Kannada characters (like: ನಮಸ್ಕಾರ). No English, no explanation.",
    "hi": "Translate the following English text to Hindi Devanagari script. Output ONLY Hindi characters (like: नमस्ते). No English, no explanation.",
}

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
    """Pass 1: speech -> English via Groq Whisper."""
    result = groq_client.audio.translations.create(
        file=("audio.wav", wav_bytes),
        model=WHISPER_MODEL,
        response_format="text",
    )
    text = result.text if hasattr(result, "text") else str(result)
    return cleanup_text(text.strip())

def _groq_llm_to_native(english_text: str, source_lang: str) -> str:
    """Pass 2: English -> native Unicode via Groq Llama."""
    if not english_text:
        return ""
    system_prompt = _NATIVE_PROMPTS.get(source_lang, _NATIVE_PROMPTS["ml"])
    print(f"[ASR] LLM call: lang={source_lang}, len={len(english_text)}")
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": english_text},
        ],
        max_tokens=1024,
        temperature=0.1,
    )
    result = response.choices[0].message.content.strip()
    print(f"[ASR] LLM result: {result[:80]}")
    return cleanup_text(result)

def transcribe_audio(audio_input, source_lang: str = "ml"):
    try:
        wav_bytes = _to_wav_bytes(audio_input)
        print(f"[ASR] lang={source_lang} bytes={len(wav_bytes)}")
        english_text = _groq_translate(wav_bytes)
        print(f"[ASR] English  : {english_text}")
        native_text = _groq_llm_to_native(english_text, source_lang)
        print(f"[ASR] Native   : {native_text}")
        return {
            "status": "success", "text": english_text,
            "malayalam_text": native_text, "raw_text": english_text,
            "language": source_lang, "source_lang": source_lang,
            "segments": [], "device": DEVICE, "model": WHISPER_MODEL,
        }
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return {"status": "failed", "error": str(e), "text": "", "malayalam_text": "", "segments": []}

def transcribe_audio_stream(audio_input, source_lang: str = "ml"):
    try:
        wav_bytes = _to_wav_bytes(audio_input)
        print(f"[ASR] stream lang={source_lang} bytes={len(wav_bytes)}")
        english_text = _groq_translate(wav_bytes)
        print(f"[ASR] English  : {english_text}")
        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}
        native_text = _groq_llm_to_native(english_text, source_lang)
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