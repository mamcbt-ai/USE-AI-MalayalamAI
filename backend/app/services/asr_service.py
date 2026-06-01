import os, re, tempfile
import numpy as np, soundfile as sf
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL     = "llama-3.3-70b-versatile"
DEVICE        = "groq-whisper+llm"

LANG_CODES = {"ml":"ml","ta":"ta","te":"te","kn":"kn","hi":"hi"}
LANG_NAMES = {"ml":"Malayalam","ta":"Tamil","te":"Telugu","kn":"Kannada","hi":"Hindi"}

HALLUCINATIONS = [
    "thank you for watching","thanks for watching","subscribe",
    "hello and welcome","welcome to my channel","translated by",
    "translation by","music","applause","subtitles","captions",
]

print(f"ASR ready: {WHISPER_MODEL} + {LLM_MODEL}")

def cleanup_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([.!?])\1+", r"\1", text)
    return text

def is_hallucination(text: str) -> bool:
    if not text or len(text.strip()) < 3: return True
    lower = text.lower().strip()
    if any(p in lower for p in HALLUCINATIONS): return True
    words = re.findall(r"\w+", lower, flags=re.UNICODE)
    if len(words) >= 6 and len(set(words)) / max(len(words),1) < 0.45: return True
    if len(lower) <= 12 and lower in {"hello","welcome","thanks","thank you"}: return True
    return False

def _load_audio(audio_input):
    if isinstance(audio_input, (bytes, bytearray)):
        import tempfile as tf2
        with tf2.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_input)
            p = tmp.name
        try:
            import av, io
            samples = []
            with av.open(p) as container:
                for frame in container.decode(audio=0):
                    arr = frame.to_ndarray()
                    if arr.ndim > 1: arr = arr.mean(axis=0)
                    samples.append(arr.astype(np.float32))
            audio = np.concatenate(samples) if samples else np.zeros(1, dtype=np.float32)
            if audio.max() > 1.5: audio /= 32768.0
        except Exception as e:
            print(f"[ASR] decode error: {e}")
            audio = np.zeros(1, dtype=np.float32)
        finally:
            os.unlink(p)
        sr = 16000
    elif isinstance(audio_input, np.ndarray):
        audio, sr = audio_input.astype("float32"), 16000
    else:
        audio, sr = sf.read(audio_input, dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        dur = len(audio) / sr
        audio = np.interp(
            np.linspace(0, dur, int(dur*16000), endpoint=False),
            np.linspace(0, dur, len(audio), endpoint=False),
            audio
        ).astype("float32")
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.98: audio = audio * (0.98 / peak)
    return audio

def _to_wav_bytes(audio_input) -> bytes:
    audio = _load_audio(audio_input)
    audio = audio[:25*16000]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        p = tmp.name
    try:
        with open(p,"rb") as f: return f.read()
    finally:
        os.unlink(p)

def _llm_call(system: str, user_text: str) -> str:
    if not user_text: return ""
    try:
        r = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user_text}],
            temperature=0.0, max_tokens=512,
        )
        return cleanup_text(r.choices[0].message.content.strip())
    except Exception as e:
        print(f"[ASR] LLM error: {e}"); return ""

def _transcribe_native(wav_bytes: bytes, lang: str) -> str:
    try:
        result = groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            language=LANG_CODES.get(lang, "ml"),
            response_format="text",
        )
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        text = cleanup_text(text)
        if is_hallucination(text): return ""
        return text
    except Exception as e:
        print(f"[ASR] transcription error: {e}"); return ""

def _refine_native_text(native_text: str, lang: str) -> str:
    if not native_text: return ""
    lang_name = LANG_NAMES.get(lang, lang)
    system = (
        f"You are an expert editor for {lang_name}. "
        f"Clean this ASR transcript in {lang_name} script. "
        "Fix obvious ASR mistakes, punctuation, spacing, and malformed words. "
        "Preserve slang, colloquial meaning, names, places, and numbers. "
        "Do NOT translate. Output ONLY the corrected text in the same language/script."
    )
    return _llm_call(system, native_text) or native_text

def _translate_to_english(native_text: str, lang: str) -> str:
    if not native_text: return ""
    lang_name = LANG_NAMES.get(lang, lang)
    system = (
        f"You are an expert {lang_name}-to-English translator. "
        f"Translate the user's {lang_name} text into natural English. "
        "Preserve names, places, slang meaning, tone, and numbers. "
        "Do not summarize. Output only English."
    )
    return _llm_call(system, native_text)

def _make_result(raw_native: str, native_text: str, english_text: str, source_lang: str) -> dict:
    return {
        "status":         "success",
        "text":           english_text,
        "english_text":   english_text,
        "raw_native_text": raw_native,
        "native_text":    native_text,
        "malayalam_text": native_text,
        "display_text":   native_text,
        "language":       source_lang,
        "language_name":  LANG_NAMES.get(source_lang, source_lang),
        "segments":       [],
        "device":         DEVICE,
        "model":          WHISPER_MODEL,
    }

def transcribe_audio(audio_input, style="standard", source_lang="ml"):
    try:
        wav = _to_wav_bytes(audio_input)
        print(f"[ASR] lang={source_lang} bytes={len(wav)}")
        native_raw  = _transcribe_native(wav, source_lang)
        print(f"[ASR] raw   = {native_raw[:120]}")
        native_text = _refine_native_text(native_raw, source_lang)
        print(f"[ASR] refined = {native_text[:120]}")
        english     = _translate_to_english(native_text or native_raw, source_lang)
        print(f"[ASR] english = {english[:120]}")
        return _make_result(native_raw, native_text, english, source_lang)
    except Exception as e:
        print(f"[ASR] error: {e}")
        return {"status":"failed","error":str(e),"text":"","english_text":"","native_text":"","display_text":"","segments":[]}

def transcribe_audio_stream(audio_input, style="standard", source_lang="ml"):
    try:
        wav = _to_wav_bytes(audio_input)
        print(f"[ASR] stream lang={source_lang} bytes={len(wav)}")
        native_raw  = _transcribe_native(wav, source_lang)
        native_text = _refine_native_text(native_raw, source_lang)
        english     = _translate_to_english(native_text or native_raw, source_lang)
        if english:
            yield {"type":"english_segment","text":english,"accumulated":english}
        if native_text:
            yield {"type":"native_segment","text":native_text,"accumulated":native_text}
            yield {"type":"malayalam_segment","text":native_text,"accumulated":native_text}
        yield {"type":"complete","english_text":english,"native_text":native_text,
               "malayalam_text":native_text,"language":source_lang,"source_lang":source_lang}
    except Exception as e:
        print(f"[ASR] stream error: {e}")
        yield {"type":"error","error":str(e)}
