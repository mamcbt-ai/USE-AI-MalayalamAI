import os, re, numpy as np, tempfile, requests, soundfile as sf
from groq import Groq

_groq = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
_SARVAM_KEY = os.environ.get("SARVAM_API_KEY", "")
_LLM_MODEL = "llama-3.3-70b-versatile"
print(f"ASR ready: Sarvam saaras:v3 (direct) + {_LLM_MODEL}")

LANGUAGES = {
    "ml": {"code": "ml-IN", "name": "Malayalam"},
    "ta": {"code": "ta-IN", "name": "Tamil"},
    "te": {"code": "te-IN", "name": "Telugu"},
    "kn": {"code": "kn-IN", "name": "Kannada"},
    "hi": {"code": "hi-IN", "name": "Hindi"},
}

_HALLUCINATIONS = [
    "thank you for watching","thanks for watching","subscribe","like and subscribe",
    "music plays","[music]","[applause]","subtitles by","www.","http",
    "hello and welcome to my channel","welcome to my youtube",
    "this video is sponsored","romantic music","background music",
]

def _translation_prompt(lang):
    name = LANGUAGES.get(lang, LANGUAGES["ml"])["name"]
    return (
        f"You are a professional {name}-to-English translator. "
        "Translate EXACTLY what is written. Preserve all names, numbers, places. "
        "Do NOT summarize or add anything. Output ONLY the English translation."
    )

def _is_bad(text):
    if not text or len(text.strip()) < 3: return True
    lower = text.lower()
    if any(h in lower for h in _HALLUCINATIONS): return True
    words = text.split()
    if len(words) >= 6 and len(set(w.lower() for w in words)) / len(words) < 0.4: return True
    return False

def _dedup(text):
    if not text: return ""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    seen, out = set(), []
    for p in parts:
        key = re.sub(r"\s+", " ", p.strip().lower())[:50]
        if key and key not in seen:
            seen.add(key)
            out.append(p.strip())
    return " ".join(out)

def _sarvam_transcribe(audio_input, lang):
    if not _SARVAM_KEY:
        print("[ASR] SARVAM_API_KEY not set")
        return ""
    lang_code = LANGUAGES.get(lang, LANGUAGES["ml"])["code"]

    # If raw bytes (WebM/WAV), send directly — best quality
    if isinstance(audio_input, (bytes, bytearray)):
        audio_bytes = audio_input
        mime = "audio/webm"
        filename = "audio.webm"
    else:
        # numpy array — convert to WAV
        audio = audio_input if isinstance(audio_input, np.ndarray) else audio_input
        if not isinstance(audio, np.ndarray):
            audio, _ = sf.read(audio_input, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        audio = audio[:25*16000]
        peak = np.max(np.abs(audio))
        if peak > 0: audio = audio * (0.891 / peak)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, 16000, subtype="PCM_16")
            p = tmp.name
        audio_bytes = open(p,"rb").read()
        os.unlink(p)
        mime = "audio/wav"
        filename = "audio.wav"

    print(f"[ASR] Sarvam ({lang}) {len(audio_bytes)} bytes ({mime})")
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": _SARVAM_KEY},
            files={"file": (filename, audio_bytes, mime)},
            data={"language_code": lang_code, "model": "saaras:v3"},
            timeout=30,
        )
        if resp.status_code == 200:
            t = resp.json().get("transcript", "").strip()
            print(f"[ASR] Sarvam result: {t[:120]}")
            if _is_bad(t):
                print("[ASR] Filtered as hallucination/repetition")
                return ""
            return _dedup(t)
        print(f"[ASR] Sarvam {resp.status_code}: {resp.text[:200]}")
        return ""
    except Exception as e:
        print(f"[ASR] Sarvam error: {e}")
        return ""

def _llm_translate(native, lang):
    if not native: return ""
    try:
        r = _groq.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role":"system","content":_translation_prompt(lang)},{"role":"user","content":native}],
            max_tokens=512, temperature=0.0)
        result = r.choices[0].message.content.strip()
        print(f"[ASR] English: {result[:120]}")
        return result
    except Exception as e:
        print(f"[ASR] LLM error: {e}")
        return ""

def transcribe_audio(audio_input, style="standard", source_lang="ml"):
    try:
        native  = _sarvam_transcribe(audio_input, source_lang)
        english = _llm_translate(native, source_lang)
        return {"status":"success","text":english,"malayalam_text":native,"raw_text":native,
                "language":source_lang,"segments":[],"device":"sarvam+groq","model":"saaras:v3"}
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return {"status":"failed","error":str(e),"text":"","malayalam_text":"","segments":[]}

def transcribe_audio_stream(audio_input, style="standard", source_lang="ml"):
    try:
        native  = _sarvam_transcribe(audio_input, source_lang)
        english = _llm_translate(native, source_lang)
        if english: yield {"type":"english_segment","text":english,"accumulated":english}
        if native:  yield {"type":"malayalam_segment","text":native,"accumulated":native}
        yield {"type":"complete","english_text":english,"malayalam_text":native,
               "language":source_lang,"source_lang":source_lang}
    except Exception as e:
        print(f"[ASR] Stream Error: {e}")
        yield {"type":"error","error":str(e)}
