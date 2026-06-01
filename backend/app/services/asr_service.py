import os, re, tempfile
from typing import Dict, Generator, Union
import numpy as np, requests, soundfile as sf
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
SARVAM_KEY  = os.environ.get("SARVAM_API_KEY", "")
LLM_MODEL   = "llama-3.3-70b-versatile"
DEVICE      = "sarvam:saaras:v3"

LANG_CODES = {"ml":"ml-IN","ta":"ta-IN","te":"te-IN","kn":"kn-IN","hi":"hi-IN"}
LANG_NAMES = {"ml":"Malayalam","ta":"Tamil","te":"Telugu","kn":"Kannada","hi":"Hindi"}

HALLUCINATIONS = [
    "thank you for watching","thanks for watching","subscribe and like",
    "hello and welcome","music plays","applause","[music]","[applause]",
    "www.","http","subtitles by","translated by","welcome to my channel",
    "new episode of the video game","romantic music plays",
]

UNICODE_ONLY_PROMPTS = {
    "ml":"You are a Malayalam expert. Convert the following transcript into clean, natural Malayalam Unicode script. Preserve names, places, and numbers exactly. Output ONLY Malayalam Unicode text.",
    "ta":"You are a Tamil expert. Convert the following transcript into clean, natural Tamil Unicode script. Preserve names, places, and numbers exactly. Output ONLY Tamil Unicode text.",
    "te":"You are a Telugu expert. Convert the following transcript into clean, natural Telugu Unicode script. Preserve names, places, and numbers exactly. Output ONLY Telugu Unicode text.",
    "kn":"You are a Kannada expert. Convert the following transcript into clean, natural Kannada Unicode script. Preserve names, places, and numbers exactly. Output ONLY Kannada Unicode text.",
    "hi":"You are a Hindi expert. Convert the following transcript into clean, natural Hindi Devanagari script. Preserve names, places, and numbers exactly. Output ONLY Hindi Unicode text.",
}

def cleanup_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\s+", " ", text.strip())
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    seen, out = set(), []
    for p in parts:
        key = p.lower()[:60]
        if key not in seen:
            seen.add(key); out.append(p)
    return " ".join(out).strip()

def _is_hallucination(text: str) -> bool:
    if not text or len(text.strip()) < 3: return False
    lower = text.lower().strip()
    if any(h in lower for h in HALLUCINATIONS): return True
    words = re.findall(r"\w+", lower)
    if len(words) >= 6 and len(set(words)) / max(len(words),1) < 0.4: return True
    return False

def _prepare_audio(audio_input) -> tuple:
    # Returns (bytes, mime_type) — sends raw bytes when possible
    if isinstance(audio_input, (bytes, bytearray)):
        return bytes(audio_input), "audio/webm"
    if isinstance(audio_input, np.ndarray):
        audio = audio_input.astype("float32")
    else:
        try:
            audio, sr = sf.read(audio_input, dtype="float32")
        except Exception:
            with open(audio_input, "rb") as f:
                return f.read(), "audio/webm"
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    audio = audio[:25*16000]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0: audio = audio * (0.95 / peak)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        p = tmp.name
    try:
        with open(p, "rb") as f: return f.read(), "audio/wav"
    finally:
        if os.path.exists(p): os.unlink(p)

def _sarvam_transcribe(audio_bytes: bytes, mime: str, lang: str) -> str:
    if not SARVAM_KEY: return ""
    lang_code = LANG_CODES.get(lang, "ml-IN")
    filename  = "audio.webm" if "webm" in mime else "audio.wav"
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_KEY},
            files={"file": (filename, audio_bytes, mime)},
            data={"language_code": lang_code, "model": "saaras:v3"},
            timeout=30,
        )
        if resp.status_code == 200:
            result = resp.json().get("transcript", "").strip()
            print(f"[ASR] Sarvam ({lang}): {result[:120]}")
            if _is_hallucination(result): return ""
            return cleanup_text(result)
        print(f"[ASR] Sarvam {resp.status_code}: {resp.text[:200]}")
        return ""
    except Exception as e:
        print(f"[ASR] Sarvam error: {e}"); return ""

def _llm_call(system_prompt: str, user_text: str) -> str:
    if not user_text: return ""
    try:
        r = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_text}],
            max_tokens=512, temperature=0.0,
        )
        return cleanup_text(r.choices[0].message.content.strip())
    except Exception as e:
        print(f"[ASR] LLM error: {e}"); return ""

def _normalize_unicode(native_text: str, lang: str) -> str:
    if not native_text: return ""
    ascii_ratio = sum(1 for c in native_text if ord(c) < 128) / max(len(native_text), 1)
    if ascii_ratio < 0.25: return native_text   # Already mostly native Unicode
    return _llm_call(UNICODE_ONLY_PROMPTS.get(lang, UNICODE_ONLY_PROMPTS["ml"]), native_text) or native_text

def _translate_to_english(native_text: str, lang: str) -> str:
    if not native_text: return ""
    lang_name = LANG_NAMES.get(lang, lang)
    system = (
        f"You are a professional {lang_name}-to-English translator. "
        "Translate EXACTLY what is written. Preserve all names, numbers, places, and meaning. "
        "Do NOT summarize, add, or remove anything. Output ONLY the English translation."
    )
    return _llm_call(system, native_text)

def transcribe_audio(audio_input, style: str = "standard", source_lang: str = "ml") -> Dict:
    try:
        audio_bytes, mime = _prepare_audio(audio_input)
        print(f"[ASR] lang={source_lang} bytes={len(audio_bytes)} mime={mime}")
        native_raw  = _sarvam_transcribe(audio_bytes, mime, source_lang)
        native_text = _normalize_unicode(native_raw, source_lang) if native_raw else ""
        english     = _translate_to_english(native_text or native_raw, source_lang)
        print(f"[ASR] Native : {native_text[:120]}")
        print(f"[ASR] English: {english[:120]}")
        return {
            "status": "success",
            "text": english, "english_text": english,
            "native_text": native_text, "malayalam_text": native_text,
            "raw_text": native_raw,
            "language": source_lang,
            "language_name": LANG_NAMES.get(source_lang, source_lang),
            "segments": [], "device": DEVICE, "model": "saaras:v3",
        }
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return {"status":"failed","error":str(e),"text":"","english_text":"","native_text":"","malayalam_text":"","segments":[]}

def transcribe_audio_stream(audio_input, style: str = "standard", source_lang: str = "ml") -> Generator[Dict, None, None]:
    try:
        audio_bytes, mime = _prepare_audio(audio_input)
        print(f"[ASR] stream lang={source_lang} bytes={len(audio_bytes)} mime={mime}")
        native_raw  = _sarvam_transcribe(audio_bytes, mime, source_lang)
        native_text = _normalize_unicode(native_raw, source_lang) if native_raw else ""
        english     = _translate_to_english(native_text or native_raw, source_lang)
        if english:     yield {"type":"english_segment",   "text":english,     "accumulated":english}
        if native_text: yield {"type":"malayalam_segment", "text":native_text, "accumulated":native_text}
        yield {"type":"complete","english_text":english,"malayalam_text":native_text,"native_text":native_text,"language":source_lang,"source_lang":source_lang}
    except Exception as e:
        print(f"[ASR] Stream Error: {e}")
        yield {"type":"error","error":str(e)}
