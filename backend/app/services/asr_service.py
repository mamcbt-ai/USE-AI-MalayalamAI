import os, re, tempfile
from typing import Dict, Generator, Optional, Tuple, Union
import numpy as np, requests, soundfile as sf
from groq import Groq

try:
    import librosa
except ImportError:
    librosa = None

groq_client      = Groq(api_key=os.environ.get("GROQ_API_KEY"))
SARVAM_KEY       = os.environ.get("SARVAM_API_KEY", "")
TARGET_SR        = 16000
MAX_SECONDS      = 25
TRANSLATION_MODEL = "llama-3.3-70b-versatile"
DEVICE           = "sarvam:saaras:v3"

LANG_CODES  = {"ml":"ml-IN","ta":"ta-IN","te":"te-IN","kn":"kn-IN","hi":"hi-IN"}
LANG_NAMES  = {"ml":"Malayalam","ta":"Tamil","te":"Telugu","kn":"Kannada","hi":"Hindi"}
UNICODE_LABELS = {"ml":"MALAYALAM UNICODE","ta":"TAMIL UNICODE","te":"TELUGU UNICODE","kn":"KANNADA UNICODE","hi":"HINDI UNICODE"}

BAD_PHRASES = [
    "thank you for watching","thanks for watching","subscribe","like and share",
    "hello and welcome","welcome to my channel","translated by","subtitles by",
    "music plays","[music]","[applause]","romantic music",
    "new episode of the video game","my all dear people","story about a little boy",
]

print(f"ASR ready: {DEVICE} + {TRANSLATION_MODEL}")

def cleanup_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return text.strip()

def _looks_like_hallucination(text: str) -> bool:
    if not text or len(text.strip()) < 2: return True
    lower = text.lower()
    if any(p in lower for p in BAD_PHRASES): return True
    words = re.findall(r"\w+", lower)
    if len(words) >= 8 and len(set(words)) / max(len(words),1) < 0.38: return True
    return False

def _native_script_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    return sum(1 for c in letters if ord(c) > 127) / len(letters)

def _load_audio(audio_input) -> Tuple[np.ndarray, int]:
    if isinstance(audio_input, (bytes, bytearray)):
        # Decode WebM/Opus via PyAV
        import tempfile as tf2
        with tf2.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_input); p = tmp.name
        try:
            import av
            samples = []
            with av.open(p) as container:
                for frame in container.decode(audio=0):
                    arr = frame.to_ndarray()
                    if arr.ndim > 1: arr = arr.mean(axis=0)
                    samples.append(arr.astype(np.float32))
            audio = np.concatenate(samples) if samples else np.zeros(1, dtype=np.float32)
            if len(audio) and np.max(np.abs(audio)) > 1.5: audio /= 32768.0
        except Exception as e:
            print(f"[ASR] decode error: {e}"); audio = np.zeros(1, dtype=np.float32)
        finally:
            os.unlink(p)
        return audio.astype("float32"), TARGET_SR
    elif isinstance(audio_input, np.ndarray):
        return audio_input.astype("float32"), TARGET_SR
    else:
        audio, sr = sf.read(audio_input, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            if librosa is None:
                # Simple resampling without librosa
                dur = len(audio) / sr
                audio = np.interp(
                    np.linspace(0, dur, int(dur*TARGET_SR), endpoint=False),
                    np.linspace(0, dur, len(audio), endpoint=False), audio
                ).astype("float32")
            else:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        return audio.astype("float32"), TARGET_SR

def _prepare_wav_bytes(audio_input) -> bytes:
    audio, sr = _load_audio(audio_input)
    audio = audio[:MAX_SECONDS * TARGET_SR]
    if len(audio) == 0: raise RuntimeError("Empty audio after decoding")
    # Validate
    if len(audio) < int(0.7 * TARGET_SR): raise RuntimeError("Audio too short. Please speak for at least 1 second.")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 0.005: raise RuntimeError("Audio too quiet. Please speak louder or hold phone closer.")
    # Normalize
    peak = float(np.max(np.abs(audio)))
    if peak > 0: audio = audio / peak * 0.95
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, TARGET_SR); p = tmp.name
    try:
        with open(p,"rb") as f: return f.read()
    finally:
        os.unlink(p)

def _sarvam(wav_bytes: bytes, lang: str, mode: str) -> str:
    if not SARVAM_KEY: return ""
    lang_code = LANG_CODES.get(lang, "ml-IN")
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_KEY},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language_code": lang_code, "model": "saaras:v3", "mode": mode},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[ASR] Sarvam {mode} {resp.status_code}: {resp.text[:120]}")
            return ""
        text = cleanup_text(resp.json().get("transcript", ""))
        print(f"[ASR] Sarvam {mode}: {text[:160]}")
        if _looks_like_hallucination(text): return ""
        return text
    except Exception as e:
        print(f"[ASR] Sarvam {mode} error: {e}"); return ""

def _pick_best(primary: str, codemix: str) -> str:
    candidates = [(t, _native_script_ratio(t) + min(len(t),120)/120) for t in [primary, codemix] if t]
    if not candidates: return ""
    best = max(candidates, key=lambda x: x[1])[0]
    print(f"[ASR] Best native: {best[:160]}")
    return best

def _translate(native_text: str, lang: str) -> str:
    if not native_text: return ""
    lang_name = LANG_NAMES.get(lang, lang)
    system = (
        f"You are an expert {lang_name}-to-English translator. "
        "Translate exactly what is written. Preserve names, slang, places, brands, and numbers. "
        "Do not summarize. Do not explain. Output only natural English."
    )
    try:
        r = groq_client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":native_text}],
            max_tokens=512, temperature=0.0,
        )
        result = cleanup_text(r.choices[0].message.content.strip())
        print(f"[ASR] English: {result[:160]}")
        return result if not _looks_like_hallucination(result) else ""
    except Exception as e:
        print(f"[ASR] Translation error: {e}"); return ""

def _ok(native: str, english: str, lang: str) -> Dict:
    return {
        "status": "success",
        "text": english, "english_text": english,
        "native_text": native, "malayalam_text": native, "raw_text": native,
        "language": lang,
        "native_language_name": LANG_NAMES.get(lang, lang),
        "unicode_label": UNICODE_LABELS.get(lang, "NATIVE UNICODE"),
        "segments": [], "device": DEVICE, "model": "saaras:v3",
    }

def _fail(msg: str, lang: str) -> Dict:
    return {
        "status": "failed", "error": msg,
        "text": "", "english_text": "", "native_text": "", "malayalam_text": "",
        "language": lang,
        "native_language_name": LANG_NAMES.get(lang, lang),
        "unicode_label": UNICODE_LABELS.get(lang, "NATIVE UNICODE"),
        "segments": [],
    }

def transcribe_audio(audio_input, style="standard", source_lang="ml") -> Dict:
    try:
        wav = _prepare_wav_bytes(audio_input)
        print(f"[ASR] lang={source_lang} bytes={len(wav)}")
        native_t  = _sarvam(wav, source_lang, "transcribe")
        native_cm = _sarvam(wav, source_lang, "codemix")
        native    = _pick_best(native_t, native_cm)
        if not native:
            return _fail("No clear speech recognized. Please speak 2-8 seconds in the selected language.", source_lang)
        english = _translate(native, source_lang)
        return _ok(native, english, source_lang)
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return _fail(str(e), source_lang)

def transcribe_audio_stream(audio_input, style="standard", source_lang="ml") -> Generator[Dict, None, None]:
    result = transcribe_audio(audio_input, style=style, source_lang=source_lang)
    if result["status"] != "success":
        yield {"type":"error","error":result.get("error","Transcription failed")}
        return
    if result.get("english_text"):
        yield {"type":"english_segment","text":result["english_text"],"accumulated":result["english_text"]}
    if result.get("native_text"):
        yield {"type":"native_segment","text":result["native_text"],"accumulated":result["native_text"]}
        yield {"type":"malayalam_segment","text":result["native_text"],"accumulated":result["native_text"]}
    yield {
        "type":"complete",
        "english_text":  result.get("english_text",""),
        "native_text":   result.get("native_text",""),
        "malayalam_text":result.get("native_text",""),
        "language":      result.get("language",source_lang),
        "unicode_label": result.get("unicode_label","NATIVE UNICODE"),
    }
