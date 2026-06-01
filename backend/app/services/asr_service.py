import os, re, tempfile
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np, requests, soundfile as sf
from groq import Groq

groq_client       = Groq(api_key=os.environ.get("GROQ_API_KEY"))
SARVAM_KEY        = os.environ.get("SARVAM_API_KEY", "")
TARGET_SR         = 16000
MAX_SECONDS       = 25
DEVICE            = "hybrid:sarvam+groq"
SARVAM_MODEL      = "saaras:v3"
GROQ_MODEL        = "whisper-large-v3"
TRANSLATION_MODEL = "llama-3.3-70b-versatile"

LANG_CODES  = {"ml":"ml-IN","ta":"ta-IN","te":"te-IN","kn":"kn-IN","hi":"hi-IN"}
LANG_NAMES  = {"ml":"Malayalam","ta":"Tamil","te":"Telugu","kn":"Kannada","hi":"Hindi"}
UNICODE_LABELS = {"ml":"MALAYALAM UNICODE","ta":"TAMIL UNICODE","te":"TELUGU UNICODE","kn":"KANNADA UNICODE","hi":"HINDI UNICODE"}

BAD_PHRASES = [
    "thank you for watching","thanks for watching","subscribe","like and share",
    "hello and welcome","welcome to my channel","subtitles by","[music]","[applause]",
]

print(f"ASR ready: {DEVICE} | primary={SARVAM_MODEL} fallback={GROQ_MODEL}")

def cleanup_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return text.strip()

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

def _repetition_ratio(text: str) -> float:
    words = _tokenize(text)
    if not words: return 1.0
    return 1.0 - (len(set(words)) / len(words))

def _max_run_ratio(text: str) -> float:
    words = _tokenize(text)
    if not words: return 1.0
    max_run, current = 1, 1
    for i in range(1, len(words)):
        current = current + 1 if words[i] == words[i-1] else 1
        max_run = max(max_run, current)
    return max_run / len(words)

def _native_script_ratio(text: str) -> float:
    chars = [c for c in text if c.isalpha()]
    if not chars: return 0.0
    return sum(1 for c in chars if ord(c) > 127) / len(chars)

def _looks_bad(text: str) -> bool:
    if not text or len(text.strip()) < 2: return True
    lower = text.lower()
    if any(p in lower for p in BAD_PHRASES): return True
    if _repetition_ratio(text) > 0.72: return True
    if _max_run_ratio(text) > 0.45: return True
    return False

def _score_candidate(text: str, lang: str, mode: str, source: str) -> float:
    if not text: return -999.0
    words = _tokenize(text)
    if not words: return -999.0
    score  = min(len(words), 18) / 18.0 * 3.0
    score += (1.0 - _repetition_ratio(text)) * 3.0
    score += (1.0 - _max_run_ratio(text)) * 3.0
    # codemix output legitimately contains English words, so reduce native-script bonus
    native_weight = 1.0 if mode == "codemix" else 2.0
    if lang in {"ml","ta","te","kn","hi"}: score += _native_script_ratio(text) * native_weight
    score += {"verbatim":0.4,"transcribe":0.3,"codemix":0.2}.get(mode, 0.0)
    if source == "groq": score += 0.1
    if _looks_bad(text): score -= 6.0
    return score

def _load_audio(audio_input) -> Tuple[np.ndarray, int]:
    if isinstance(audio_input, (bytes, bytearray)):
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
            dur = len(audio) / sr
            audio = np.interp(
                np.linspace(0, dur, int(dur*TARGET_SR), endpoint=False),
                np.linspace(0, dur, len(audio), endpoint=False), audio
            ).astype("float32")
        return audio.astype("float32"), TARGET_SR

def _prepare_wav_bytes(audio_input) -> bytes:
    audio, sr = _load_audio(audio_input)
    audio = audio[:MAX_SECONDS * TARGET_SR]
    if len(audio) == 0: raise RuntimeError("Empty audio")
    if len(audio) < int(1.0 * TARGET_SR): raise RuntimeError("Audio too short. Please speak for at least 1 second.")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 0.008: raise RuntimeError("Audio too quiet. Please speak louder and closer to the mic.")
    peak = float(np.max(np.abs(audio)))
    if peak > 0: audio = audio / peak * 0.95
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, TARGET_SR); p = tmp.name
    try:
        with open(p,"rb") as f: return f.read()
    finally:
        os.unlink(p)

def _sarvam_call(wav_bytes: bytes, lang: str, mode: str) -> str:
    if not SARVAM_KEY: return ""
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_KEY},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language_code": LANG_CODES.get(lang,"ml-IN"), "model": SARVAM_MODEL, "mode": mode},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[ASR] Sarvam {mode} {resp.status_code}: {resp.text[:120]}")
            return ""
        text = cleanup_text(resp.json().get("transcript",""))
        print(f"[ASR] Sarvam {mode}: {text[:160]}")
        return text
    except Exception as e:
        print(f"[ASR] Sarvam {mode} error: {e}"); return ""

def _groq_fallback(wav_bytes: bytes, lang: str) -> str:
    try:
        result = groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes), model=GROQ_MODEL, language=lang,
            response_format="text",
            prompt=f"This is {LANG_NAMES.get(lang,lang)} speech. Return accurate transcript in original language/script. Preserve slang, names, numbers.",
        )
        text = cleanup_text(result.text if hasattr(result,"text") else str(result))
        print(f"[ASR] Groq fallback: {text[:160]}")
        return text
    except Exception as e:
        print(f"[ASR] Groq fallback error: {e}"); return ""

def _select_best_native(wav_bytes: bytes, lang: str) -> Tuple[str, str]:
    candidates = []
    for mode in ["transcribe", "verbatim", "codemix"]:
        text = _sarvam_call(wav_bytes, lang, mode)
        if text:
            score = _score_candidate(text, lang, mode, "sarvam")
            candidates.append({"source":"sarvam","mode":mode,"text":text,"score":score})

    # Use Groq as fallback if best Sarvam is low-scoring OR obviously bad
    sarvam_winner = max(candidates, key=lambda x: x["score"]) if candidates else None
    need_fallback = (
        sarvam_winner is None or
        sarvam_winner["score"] < 2.5 or
        _looks_bad(sarvam_winner["text"])
    )
    if need_fallback:
        groq_text = _groq_fallback(wav_bytes, lang)
        if groq_text:
            score = _score_candidate(groq_text, lang, "transcribe", "groq")
            candidates.append({"source":"groq","mode":"transcribe","text":groq_text,"score":score})

    if not candidates: return "", ""
    candidates.sort(key=lambda x: x["score"], reverse=True)
    print("[ASR] Candidates:")
    for c in candidates:
        print(f"  {c['source']}:{c['mode']} score={c['score']:.2f} | {c['text'][:80]}")
    best = candidates[0]
    if best["score"] < 1.5: return "", ""
    return best["text"], f"{best['source']}:{best['mode']}"

def _translate_to_english(native_text: str, lang: str) -> str:
    if not native_text: return ""
    lang_name = LANG_NAMES.get(lang, lang)
    system = (
        f"You are an expert {lang_name}-to-English translator. "
        "Translate exactly what is written. Preserve names, slang, places, numbers. "
        "Do not summarize. Output only English."
    )
    try:
        r = groq_client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":native_text}],
            max_tokens=512, temperature=0.0,
        )
        result = cleanup_text(r.choices[0].message.content.strip())
        print(f"[ASR] English: {result[:160]}")
        return result
    except Exception as e:
        print(f"[ASR] Translation error: {e}"); return ""

def _success_response(native: str, english: str, lang: str, path: str) -> Dict:
    return {
        "status": "success",
        "text": english, "english_text": english,
        "native_text": native, "malayalam_text": native, "raw_text": native,
        "language": lang, "native_language": lang,
        "native_language_name": LANG_NAMES.get(lang, lang),
        "unicode_label": UNICODE_LABELS.get(lang, "NATIVE UNICODE"),
        "segments": [], "device": DEVICE, "model": path,
    }

def _failed_response(msg: str, lang: str) -> Dict:
    return {
        "status":"failed","error":msg,"text":"","english_text":"","native_text":"","malayalam_text":"",
        "language":lang,"native_language":lang,
        "native_language_name": LANG_NAMES.get(lang,lang),
        "unicode_label": UNICODE_LABELS.get(lang,"NATIVE UNICODE"),
        "segments":[],
    }

def transcribe_audio(audio_input, style="standard", source_lang="ml") -> Dict:
    try:
        wav = _prepare_wav_bytes(audio_input)
        print(f"[ASR] lang={source_lang} bytes={len(wav)}")
        native, path = _select_best_native(wav, source_lang)
        if not native:
            return _failed_response("Could not recognize clear speech. Please speak 2-8 seconds in the selected language.", source_lang)
        english = _translate_to_english(native, source_lang)
        return _success_response(native, english, source_lang, path)
    except Exception as e:
        print(f"[ASR] Error: {e}")
        return _failed_response(str(e), source_lang)

def transcribe_audio_stream(audio_input, style="standard", source_lang="ml") -> Generator[Dict, None, None]:
    result = transcribe_audio(audio_input, style=style, source_lang=source_lang)
    if result["status"] != "success":
        yield {"type":"error","error":result.get("error","Transcription failed")}; return
    if result.get("english_text"):
        yield {"type":"english_segment","text":result["english_text"],"accumulated":result["english_text"]}
    if result.get("native_text"):
        yield {"type":"native_segment","text":result["native_text"],"accumulated":result["native_text"]}
        yield {"type":"malayalam_segment","text":result["native_text"],"accumulated":result["native_text"]}
    yield {
        "type":"complete",
        "english_text": result.get("english_text",""),
        "native_text":  result.get("native_text",""),
        "malayalam_text":result.get("native_text",""),
        "language":     result.get("language",source_lang),
        "unicode_label":result.get("unicode_label","NATIVE UNICODE"),
    }
