import os, re, tempfile
from typing import Dict, Generator, List, Optional, Tuple, Union
import numpy as np, requests, soundfile as sf
from groq import Groq

TARGET_SR            = 16000
MAX_SECONDS          = 25
SARVAM_MODEL         = "saaras:v3"
GROQ_STT_MODEL       = "whisper-large-v3"
GROQ_TRANSLATE_MODEL = "llama-3.3-70b-versatile"
DEVICE               = "hybrid:sarvam+groq"
SARVAM_KEY           = os.environ.get("SARVAM_API_KEY", "").strip()

LANG_CODES = {"ml":"ml-IN","ta":"ta-IN","te":"te-IN","kn":"kn-IN","hi":"hi-IN","en":"en-IN","auto":"unknown"}
LANG_NAMES = {"ml":"Malayalam","ta":"Tamil","te":"Telugu","kn":"Kannada","hi":"Hindi","en":"English","auto":"Auto-detect"}
UNICODE_LABELS = {"ml":"MALAYALAM UNICODE","ta":"TAMIL UNICODE","te":"TELUGU UNICODE","kn":"KANNADA UNICODE","hi":"HINDI UNICODE","en":"ENGLISH TEXT","auto":"NATIVE TEXT"}
HALLUCINATION_PHRASES = ["thank you for watching","thanks for watching","subscribe","like and share","hello and welcome","welcome back to the channel","subtitles by","[music]","[applause]"]

_groq_client = None
print(f"ASR ready: {DEVICE} | primary={SARVAM_MODEL} | fallback={GROQ_STT_MODEL}")

def _get_groq() -> Optional[Groq]:
    global _groq_client
    if _groq_client is None:
        key = os.environ.get("GROQ_API_KEY","").strip()
        if not key: return None
        _groq_client = Groq(api_key=key)
    return _groq_client

def cleanup_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"\s+"," ",text.strip())
    text = re.sub(r"\s+([,.;!?])",r"\1",text)
    return text.strip()

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+",(text or "").lower(),flags=re.UNICODE)

def _contains_ascii_words(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}",text or ""))

def _repetition_ratio(text: str) -> float:
    words = _tokenize(text)
    if not words: return 1.0
    return 1.0 - (len(set(words)) / len(words))

def _max_run_ratio(text: str) -> float:
    words = _tokenize(text)
    if not words: return 1.0
    longest, current = 1, 1
    for i in range(1, len(words)):
        current = current + 1 if words[i] == words[i-1] else 1
        longest = max(longest, current)
    return longest / len(words)

def _native_script_ratio(text: str) -> float:
    chars = [c for c in (text or "") if c.isalpha()]
    if not chars: return 0.0
    return sum(1 for c in chars if ord(c) > 127) / len(chars)

def _looks_bad(text: str) -> bool:
    if not text or len(text.strip()) < 2: return True
    lower = text.lower()
    if any(p in lower for p in HALLUCINATION_PHRASES): return True
    if _repetition_ratio(text) > 0.72: return True
    if _max_run_ratio(text) > 0.45: return True
    words = _tokenize(text)
    if len(words) >= 6 and len(set(words)) <= 2: return True
    return False

def _score_candidate(text: str, lang: str, mode: str, source: str) -> float:
    if not text: return -999.0
    words = _tokenize(text)
    if not words: return -999.0
    score  = min(len(words), 18) / 18.0 * 3.0
    score += (1.0 - _repetition_ratio(text)) * 3.0
    score += (1.0 - _max_run_ratio(text)) * 3.0
    script_bonus = _native_script_ratio(text) * 2.0
    if mode == "codemix": script_bonus *= 0.35
    score += script_bonus
    if mode == "codemix":
        score += 0.55
        if _contains_ascii_words(text): score += 0.35
    elif mode == "verbatim":   score += 0.30
    elif mode == "transcribe": score += 0.20
    if source == "groq": score += 0.15
    if _looks_bad(text): score -= 6.0
    return score

def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr: return audio.astype("float32")
    if len(audio) == 0: return np.zeros(0, dtype=np.float32)
    dur = len(audio) / float(orig_sr)
    return np.interp(
        np.linspace(0, dur, max(1, int(dur*target_sr)), endpoint=False),
        np.linspace(0, dur, len(audio), endpoint=False), audio
    ).astype("float32")

def _decode_bytes_audio(audio_input: Union[bytes, bytearray]) -> Tuple[np.ndarray, int]:
    try:
        import av
    except Exception as e:
        raise RuntimeError(f"PyAV required for audio decoding. Install 'av'. Error: {e}")
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_input); tmp_path = tmp.name
    try:
        chunks = []; sample_rate = TARGET_SR
        with av.open(tmp_path) as container:
            if not container.streams.audio: raise RuntimeError("No audio stream found")
            stream = container.streams.audio[0]
            if getattr(stream.codec_context, "sample_rate", None):
                sample_rate = int(stream.codec_context.sample_rate)
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()
                if arr.ndim == 2: arr = arr.mean(axis=0)
                chunks.append(arr.astype(np.float32))
                if getattr(frame, "sample_rate", None):
                    sample_rate = int(frame.sample_rate)
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if len(audio) and np.max(np.abs(audio)) > 1.5: audio = audio / 32768.0
        return audio.astype("float32"), sample_rate
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

def _load_audio(audio_input) -> Tuple[np.ndarray, int]:
    if isinstance(audio_input, (bytes, bytearray)): return _decode_bytes_audio(audio_input)
    if isinstance(audio_input, np.ndarray): return audio_input.astype("float32"), TARGET_SR
    audio, sr = sf.read(audio_input, dtype="float32")
    if audio.ndim > 1: audio = audio.mean(axis=1)
    return audio.astype("float32"), sr

def _prepare_wav_bytes(audio_input) -> bytes:
    audio, sr = _load_audio(audio_input)
    if sr != TARGET_SR: audio = _resample_linear(audio, sr, TARGET_SR)
    audio = audio[:MAX_SECONDS * TARGET_SR]
    if len(audio) == 0: raise RuntimeError("Empty audio")
    if len(audio) < int(1.0 * TARGET_SR): raise RuntimeError("Audio too short. Please speak at least 1 second.")
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 0.008: raise RuntimeError("Audio too quiet. Please speak louder and closer to the mic.")
    peak = float(np.max(np.abs(audio)))
    if peak > 0: audio = audio / peak * 0.95
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, TARGET_SR); wav_path = tmp.name
    try:
        with open(wav_path, "rb") as f: return f.read()
    finally:
        if os.path.exists(wav_path): os.unlink(wav_path)

def _sarvam_call(wav_bytes: bytes, lang: str, mode: str) -> Tuple[str, Optional[str], Optional[float]]:
    if not SARVAM_KEY: return "", None, None
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_KEY},
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"language_code": LANG_CODES.get(lang, "unknown"), "model": SARVAM_MODEL, "mode": mode},
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[ASR] Sarvam {mode} {resp.status_code}: {resp.text[:200]}")
            return "", None, None
        payload = resp.json()
        text = cleanup_text(payload.get("transcript", ""))
        detected_lang = payload.get("language_code") or payload.get("language")
        detected_conf = payload.get("language_confidence")
        if detected_lang: print(f"[ASR] Sarvam detected={detected_lang} conf={detected_conf}")
        print(f"[ASR] Sarvam {mode}: {text[:180]}")
        return text, detected_lang, detected_conf
    except Exception as e:
        print(f"[ASR] Sarvam {mode} error: {e}"); return "", None, None

def _groq_fallback(wav_bytes: bytes, lang: str) -> str:
    client = _get_groq()
    if client is None: print("[ASR] Groq fallback skipped: GROQ_API_KEY missing"); return ""
    try:
        lang_hint = None if lang == "auto" else lang
        result = client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes), model=GROQ_STT_MODEL,
            language=lang_hint, response_format="text", temperature=0.0,
            prompt=(f"This is {LANG_NAMES.get(lang, lang)} speech. Return accurate transcript in original language/script. Preserve slang, names, places, brands, numbers. If code-mixed, preserve naturally."),
        )
        text = cleanup_text(result.text if hasattr(result, "text") else str(result))
        print(f"[ASR] Groq fallback: {text[:180]}")
        return text
    except Exception as e:
        print(f"[ASR] Groq fallback error: {e}"); return ""

def _normalize_detected_lang(detected_lang: Optional[str], requested_lang: str) -> str:
    if requested_lang != "auto": return requested_lang
    if not detected_lang: return "auto"
    mapping = {"ml-IN":"ml","ta-IN":"ta","te-IN":"te","kn-IN":"kn","hi-IN":"hi","en-IN":"en"}
    return mapping.get(detected_lang, "auto")

def _select_best_native(wav_bytes: bytes, lang: str) -> Tuple[str, str, str]:
    candidates = []; best_detected = "auto"
    for mode in ["codemix", "verbatim", "transcribe"]:
        text, detected_lang, detected_conf = _sarvam_call(wav_bytes, lang, mode)
        if text:
            normalized = _normalize_detected_lang(detected_lang, lang)
            if best_detected == "auto" and normalized != "auto": best_detected = normalized
            score = _score_candidate(text, lang, mode, "sarvam")
            if detected_conf is not None and isinstance(detected_conf, (int, float)):
                score += float(detected_conf)
            candidates.append({"source":"sarvam","mode":mode,"text":text,"score":score,"detected_lang":normalized})
    best_sarvam = max(candidates, key=lambda x: x["score"], default=None)
    need_groq = (best_sarvam is None or best_sarvam["score"] < 2.5 or _looks_bad(best_sarvam["text"]))
    if need_groq:
        groq_text = _groq_fallback(wav_bytes, lang)
        if groq_text:
            candidates.append({"source":"groq","mode":"transcribe","text":groq_text,
                                "score":_score_candidate(groq_text,lang,"transcribe","groq"),"detected_lang":best_detected})
    if not candidates: return "", "", best_detected
    candidates.sort(key=lambda x: x["score"], reverse=True)
    print("[ASR] Candidates:")
    for c in candidates: print(f"  {c['source']}:{c['mode']} score={c['score']:.2f} lang={c['detected_lang']} | {c['text'][:100]}")
    best = candidates[0]
    if best["score"] < 1.5: return "", "", best_detected
    final_lang = best["detected_lang"] if lang == "auto" and best["detected_lang"] != "auto" else lang
    print(f"[ASR] Best: {best['text'][:180]}")
    return best["text"], f"{best['source']}:{best['mode']}", final_lang

def _translate_to_english(native_text: str, lang: str) -> str:
    if not native_text: return ""
    client = _get_groq()
    if client is None: print("[ASR] Translation skipped: GROQ_API_KEY missing"); return ""
    lang_name = LANG_NAMES.get(lang, lang)
    system = (f"You are a professional {lang_name}-to-English translator. Translate exactly what is written. Preserve names, places, slang, brand names, numbers. Do not summarize. If input is already English, return it unchanged. Output only the English translation.")
    try:
        r = client.chat.completions.create(
            model=GROQ_TRANSLATE_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":native_text}],
            max_tokens=512, temperature=0.0,
        )
        result = cleanup_text(r.choices[0].message.content.strip())
        print(f"[ASR] English: {result[:180]}")
        return result
    except Exception as e:
        print(f"[ASR] Translation error: {e}"); return ""

def _success_response(native: str, english: str, lang: str, path: str) -> Dict:
    return {"status":"success","text":english,"english_text":english,"native_text":native,
            "malayalam_text":native,"raw_text":native,"language":lang,"native_language":lang,
            "native_language_name":LANG_NAMES.get(lang,lang),"unicode_label":UNICODE_LABELS.get(lang,"NATIVE TEXT"),
            "segments":[],"device":DEVICE,"model":path}

def _failed_response(msg: str, lang: str) -> Dict:
    return {"status":"failed","error":msg,"text":"","english_text":"","native_text":"","malayalam_text":"",
            "language":lang,"native_language":lang,"native_language_name":LANG_NAMES.get(lang,lang),
            "unicode_label":UNICODE_LABELS.get(lang,"NATIVE TEXT"),"segments":[]}

def transcribe_audio(audio_input, style="standard", source_lang="ml") -> Dict:
    try:
        lang = source_lang if source_lang in LANG_CODES else "auto"
        wav  = _prepare_wav_bytes(audio_input)
        print(f"[ASR] lang={lang} bytes={len(wav)}")
        native, path, final_lang = _select_best_native(wav, lang)
        if not native: return _failed_response("Could not recognize clear speech. Please speak 2-8 seconds in the selected language.", lang)
        english = _translate_to_english(native, final_lang if final_lang in LANG_NAMES else lang)
        if not english and re.search(r"[A-Za-z]{3,}", native): english = native
        return _success_response(native, english, final_lang, path)
    except Exception as e:
        print(f"[ASR] Error: {e}"); return _failed_response(str(e), source_lang)

def transcribe_audio_stream(audio_input, style="standard", source_lang="ml") -> Generator[Dict, None, None]:
    try:
        result = transcribe_audio(audio_input, style=style, source_lang=source_lang)
        if result["status"] != "success":
            yield {"type":"error","error":result.get("error","Transcription failed")}; return
        if result.get("english_text"):
            yield {"type":"english_segment","text":result["english_text"],"accumulated":result["english_text"]}
        if result.get("native_text"):
            yield {"type":"native_segment","text":result["native_text"],"accumulated":result["native_text"]}
            yield {"type":"malayalam_segment","text":result["native_text"],"accumulated":result["native_text"]}
        yield {"type":"complete","english_text":result.get("english_text",""),
               "native_text":result.get("native_text",""),"malayalam_text":result.get("native_text",""),
               "language":result.get("language",source_lang),"unicode_label":result.get("unicode_label","NATIVE TEXT")}
    except Exception as e:
        print(f"[ASR] Stream Error: {e}"); yield {"type":"error","error":str(e)}
