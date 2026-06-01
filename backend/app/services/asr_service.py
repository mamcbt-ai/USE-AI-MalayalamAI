import os, re, tempfile
from typing import Dict, Generator, List, Tuple, Union
import numpy as np, requests, soundfile as sf
from groq import Groq

TARGET_SR         = 16000
MAX_SECONDS       = 25
DEVICE            = "hybrid:sarvam+groq"
SARVAM_MODEL      = "saaras:v3"
GROQ_MODEL        = "whisper-large-v3"
TRANSLATION_MODEL = "llama-3.3-70b-versatile"
SARVAM_KEY        = os.environ.get("SARVAM_API_KEY", "")

LANG_CODES  = {"ml":"ml-IN","ta":"ta-IN","te":"te-IN","kn":"kn-IN","hi":"hi-IN","en":"en-IN","auto":"unknown"}
LANG_NAMES  = {"ml":"Malayalam","ta":"Tamil","te":"Telugu","kn":"Kannada","hi":"Hindi","en":"English","auto":"Auto"}
UNICODE_LABELS = {"ml":"MALAYALAM UNICODE","ta":"TAMIL UNICODE","te":"TELUGU UNICODE","kn":"KANNADA UNICODE","hi":"HINDI UNICODE","en":"ENGLISH TEXT","auto":"NATIVE TEXT"}
BAD_PHRASES = ["thank you for watching","thanks for watching","subscribe","like and share","hello and welcome","welcome to my channel","subtitles by","[music]","[applause]"]

_groq_client = None
print(f"ASR ready: {DEVICE} | primary={SARVAM_MODEL} fallback={GROQ_MODEL}")

def _get_groq():
    global _groq_client
    if _groq_client is None:
        key = os.environ.get("GROQ_API_KEY","").strip()
        if not key: return None
        _groq_client = Groq(api_key=key)
    return _groq_client

def cleanup_text(t):
    if not t: return ""
    t = re.sub(r"\s+"," ",t.strip()); t = re.sub(r"\s+([,.;!?])",r"\1",t)
    return t.strip()

def _tokenize(t): return re.findall(r"\w+",t.lower(),flags=re.UNICODE)
def _contains_ascii_words(t): return bool(re.search(r"[A-Za-z]{2,}",t or ""))

def _repetition_ratio(t):
    w=_tokenize(t); return 1.0 if not w else 1.0-(len(set(w))/len(w))

def _max_run_ratio(t):
    w=_tokenize(t)
    if not w: return 1.0
    mx,cur=1,1
    for i in range(1,len(w)):
        cur = cur+1 if w[i]==w[i-1] else 1; mx=max(mx,cur)
    return mx/len(w)

def _native_script_ratio(t):
    ch=[c for c in t if c.isalpha()]
    return 0.0 if not ch else sum(1 for c in ch if ord(c)>127)/len(ch)

def _looks_bad(t):
    if not t or len(t.strip())<2: return True
    lower=t.lower()
    if any(p in lower for p in BAD_PHRASES): return True
    if _repetition_ratio(t)>0.72: return True
    if _max_run_ratio(t)>0.45: return True
    w=_tokenize(t)
    if len(w)>=6 and len(set(w))<=2: return True
    return False

def _score_candidate(t,lang,mode,source):
    if not t: return -999.0
    w=_tokenize(t)
    if not w: return -999.0
    score  = min(len(w),18)/18.0*3.0
    score += (1.0-_repetition_ratio(t))*3.0
    score += (1.0-_max_run_ratio(t))*3.0
    nb = _native_script_ratio(t)*2.0
    if mode=="codemix": nb*=0.35
    score += nb
    if mode=="codemix":
        score+=0.55
        if _contains_ascii_words(t): score+=0.35
    elif mode=="verbatim":   score+=0.30
    elif mode=="transcribe": score+=0.20
    if source=="groq": score+=0.15
    if _looks_bad(t): score-=6.0
    return score

def _resample_linear(audio,orig_sr,target_sr):
    if orig_sr==target_sr: return audio.astype("float32")
    if len(audio)==0: return np.zeros(0,dtype="float32")
    dur=len(audio)/orig_sr
    return np.interp(
        np.linspace(0,dur,max(1,int(dur*target_sr)),endpoint=False),
        np.linspace(0,dur,len(audio),endpoint=False),audio
    ).astype("float32")

def _decode_bytes_audio(data):
    try:
        import av
        with tempfile.NamedTemporaryFile(suffix=".webm",delete=False) as tmp:
            tmp.write(data); p=tmp.name
        try:
            frames=[]; sample_rate=TARGET_SR
            with av.open(p) as container:
                stream=container.streams.audio[0]
                if getattr(stream.codec_context,"sample_rate",None):
                    sample_rate=int(stream.codec_context.sample_rate)
                for frame in container.decode(audio=0):
                    arr=frame.to_ndarray()
                    if arr.ndim==2: arr=arr.mean(axis=0)
                    frames.append(arr.astype(np.float32))
                    if getattr(frame,"sample_rate",None):
                        sample_rate=int(frame.sample_rate)
            audio=np.concatenate(frames) if frames else np.zeros(0,dtype=np.float32)
            if len(audio) and np.max(np.abs(audio))>1.5: audio/=32768.0
            return audio.astype("float32"), sample_rate
        finally:
            if os.path.exists(p): os.unlink(p)
    except Exception as e:
        raise RuntimeError(f"Audio decode failed: {e}")

def _load_audio(audio_input):
    if isinstance(audio_input,(bytes,bytearray)): return _decode_bytes_audio(audio_input)
    if isinstance(audio_input,np.ndarray): return audio_input.astype("float32"),TARGET_SR
    audio,sr=sf.read(audio_input,dtype="float32")
    if audio.ndim>1: audio=audio.mean(axis=1)
    return audio.astype("float32"),sr

def _prepare_wav_bytes(audio_input):
    audio,sr=_load_audio(audio_input)
    if sr!=TARGET_SR: audio=_resample_linear(audio,sr,TARGET_SR)
    audio=audio[:MAX_SECONDS*TARGET_SR]
    if len(audio)==0: raise RuntimeError("Empty audio")
    if len(audio)<int(1.0*TARGET_SR): raise RuntimeError("Audio too short. Please speak at least 1 second.")
    rms=float(np.sqrt(np.mean(np.square(audio))))
    if rms<0.008: raise RuntimeError("Audio too quiet. Please speak louder.")
    peak=float(np.max(np.abs(audio)))
    if peak>0: audio=audio/peak*0.95
    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tmp:
        sf.write(tmp.name,audio,TARGET_SR); p=tmp.name
    try:
        with open(p,"rb") as f: return f.read()
    finally:
        if os.path.exists(p): os.unlink(p)

def _sarvam_call(wav,lang,mode):
    if not SARVAM_KEY: return ""
    try:
        resp=requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key":SARVAM_KEY},
            files={"file":("audio.wav",wav,"audio/wav")},
            data={"language_code":LANG_CODES.get(lang,"unknown"),"model":SARVAM_MODEL,"mode":mode},
            timeout=30,
        )
        if resp.status_code!=200: print(f"[ASR] Sarvam {mode} {resp.status_code}: {resp.text[:200]}"); return ""
        t=cleanup_text(resp.json().get("transcript",""))
        print(f"[ASR] Sarvam {mode}: {t[:180]}")
        return t
    except Exception as e:
        print(f"[ASR] Sarvam {mode} error: {e}"); return ""

def _groq_fallback(wav,lang):
    client=_get_groq()
    if client is None: print("[ASR] Groq fallback skipped: no GROQ_API_KEY"); return ""
    try:
        lang_hint=None if lang=="auto" else lang
        result=client.audio.transcriptions.create(
            file=("audio.wav",wav),model=GROQ_MODEL,language=lang_hint,response_format="text",
            prompt=(f"This is {LANG_NAMES.get(lang,lang)} speech. Return accurate transcript in original language/script. Preserve slang, names, places, numbers. If code-mixed, preserve naturally."),
        )
        t=cleanup_text(result.text if hasattr(result,"text") else str(result))
        print(f"[ASR] Groq fallback: {t[:180]}")
        return t
    except Exception as e:
        print(f"[ASR] Groq fallback error: {e}"); return ""

def _select_best_native(wav,lang):
    candidates=[]
    for mode in ["codemix","verbatim","transcribe"]:
        t=_sarvam_call(wav,lang,mode)
        if t: candidates.append({"source":"sarvam","mode":mode,"text":t,"score":_score_candidate(t,lang,mode,"sarvam")})
    best_sarvam=max(candidates,key=lambda c:c["score"],default=None)
    need_fallback=(best_sarvam is None or best_sarvam["score"]<2.5 or _looks_bad(best_sarvam["text"]))
    if need_fallback:
        gt=_groq_fallback(wav,lang)
        if gt: candidates.append({"source":"groq","mode":"transcribe","text":gt,"score":_score_candidate(gt,lang,"transcribe","groq")})
    if not candidates: return "",""
    candidates.sort(key=lambda x:x["score"],reverse=True)
    print("[ASR] Candidates:")
    for c in candidates: print(f"  {c['source']}:{c['mode']} score={c['score']:.2f} | {c['text'][:100]}")
    best=candidates[0]
    if best["score"]<1.5: return "",""
    print(f"[ASR] Best: {best['text'][:180]}")
    return best["text"],f"{best['source']}:{best['mode']}"

def _translate_to_english(native,lang):
    if not native: return ""
    client=_get_groq()
    if client is None: print("[ASR] Translation skipped: no GROQ_API_KEY"); return ""
    lang_name=LANG_NAMES.get(lang,lang)
    system=(f"You are an expert {lang_name}-to-English translator. Translate exactly what is written. Preserve names, slang, places, product names, numbers. If input is already English, return it unchanged. Do not summarize. Output only English.")
    try:
        r=client.chat.completions.create(model=TRANSLATION_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":native}],
            max_tokens=512,temperature=0.0)
        result=cleanup_text(r.choices[0].message.content.strip())
        print(f"[ASR] English: {result[:180]}")
        return result
    except Exception as e:
        print(f"[ASR] Translation error: {e}"); return ""

def _success_response(native,english,lang,path):
    return {"status":"success","text":english,"english_text":english,"native_text":native,
            "malayalam_text":native,"raw_text":native,"language":lang,"native_language":lang,
            "native_language_name":LANG_NAMES.get(lang,lang),"unicode_label":UNICODE_LABELS.get(lang,"NATIVE TEXT"),
            "segments":[],"device":DEVICE,"model":path}

def _failed_response(msg,lang):
    return {"status":"failed","error":msg,"text":"","english_text":"","native_text":"","malayalam_text":"",
            "language":lang,"native_language":lang,"native_language_name":LANG_NAMES.get(lang,lang),
            "unicode_label":UNICODE_LABELS.get(lang,"NATIVE TEXT"),"segments":[]}

def transcribe_audio(audio_input,style="standard",source_lang="ml"):
    try:
        lang=source_lang if source_lang in LANG_CODES else "auto"
        wav=_prepare_wav_bytes(audio_input)
        print(f"[ASR] lang={lang} bytes={len(wav)}")
        native,path=_select_best_native(wav,lang)
        if not native: return _failed_response("Could not recognize clear speech. Please speak 2-8 seconds in the selected language.",lang)
        english=_translate_to_english(native,lang)
        if not english and re.search(r"[A-Za-z]{3,}",native): english=native
        return _success_response(native,english,lang,path)
    except Exception as e:
        print(f"[ASR] Error: {e}"); return _failed_response(str(e),source_lang)

def transcribe_audio_stream(audio_input,style="standard",source_lang="ml"):
    try:
        result=transcribe_audio(audio_input,style=style,source_lang=source_lang)
        if result["status"]!="success":
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
