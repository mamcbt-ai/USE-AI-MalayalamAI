import os
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

NATIVE_PROMPTS = {
    "ml": """You are an expert Malayalam translator. Translate the given English text into natural, conversational Malayalam Unicode script.
Rules:
- Output ONLY Malayalam Unicode characters
- Use natural spoken Malayalam (not overly formal)
- Include common slang and colloquial expressions where appropriate
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "ഹലോ, എങ്ങനെ ഉണ്ട്?"
Translate now:""",
    "ta": """You are an expert Tamil translator. Translate the given English text into natural, conversational Tamil Unicode script.
Rules:
- Output ONLY Tamil Unicode characters
- Use natural spoken Tamil including colloquial expressions
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "வணக்கம், எப்படி இருக்கீங்க?"
Translate now:""",
    "te": """You are an expert Telugu translator. Translate the given English text into natural, conversational Telugu Unicode script.
Rules:
- Output ONLY Telugu Unicode characters
- Use natural spoken Telugu including colloquial expressions
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "హలో, ఎలా ఉన్నారు?"
Translate now:""",
    "kn": """You are an expert Kannada translator. Translate the given English text into natural, conversational Kannada Unicode script.
Rules:
- Output ONLY Kannada Unicode characters
- Use natural spoken Kannada including colloquial expressions
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "ಹಲೋ, ಹೇಗಿದ್ದೀರಾ?"
Translate now:""",
    "hi": """You are an expert Hindi translator. Translate the given English text into natural, conversational Hindi Devanagari script.
Rules:
- Output ONLY Hindi Devanagari characters
- Use natural spoken Hindi including colloquial expressions
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "नमस्ते, कैसे हो?"
Translate now:""",
}

HALLUCINATIONS = [
    "thank you for watching", "thanks for watching", "subscribe",
    "music", "[music]", "subtitles", "captions", "thank you",
    "hello and welcome", "welcome to my channel", "hello, welcome",
    "translated by", "translation by", "english is a language",
    "language of the language",
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

def _groq_translate_to_english(wav_bytes: bytes) -> str:
    """Whisper large-v3 translations -> English"""
    try:
        result = groq_client.audio.translations.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            response_format="text",
        )
        text = result.text if hasattr(result, "text") else str(result)
        text = cleanup_text(text.strip())
        if _is_hallucination(text):
            return ""
        return text
    except Exception as e:
        print(f"[ASR] Translate error: {e}")
        return ""

def _llm_to_native(english_text: str, lang: str) -> str:
    """Groq Llama: English -> native Unicode script"""
    if not english_text or lang not in NATIVE_PROMPTS:
        return ""
    try:
        resp = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": NATIVE_PROMPTS[lang]},
                {"role": "user", "content": english_text},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        result = resp.choices[0].message.content.strip()
        print(f"[ASR] LLM native ({lang}): {result[:80]}")
        return cleanup_text(result)
    except Exception as e:
        print(f"[ASR] LLM error: {e}")
        return ""

def _load_audio(audio_input) -> np.ndarray:
    if isinstance(audio_input, np.ndarray):
        return audio_input
    audio, _ = sf.read(audio_input, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio

def transcribe_audio(audio_input, style: str = "standard", source_lang: str = "ml") -> dict:
    try:
        wav_bytes = _to_wav_bytes(_load_audio(audio_input))
        print(f"[ASR] lang={source_lang} bytes={len(wav_bytes)}")

        english_text = _groq_translate_to_english(wav_bytes)
        print(f"[ASR] English: {english_text}")

        native_text = _llm_to_native(english_text, source_lang)
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

        english_text = _groq_translate_to_english(wav_bytes)
        print(f"[ASR] English: {english_text}")

        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}

        native_text = _llm_to_native(english_text, source_lang)
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
