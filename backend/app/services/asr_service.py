from faster_whisper import WhisperModel
import torch
import re
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "medium"

print(f"Loading Whisper model ({MODEL_NAME}) on {DEVICE}...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
print(f"Whisper model loaded successfully ({MODEL_NAME}/{DEVICE})")

_COMMON = dict(
    language="ml",
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 400, "threshold": 0.3},
    beam_size=5,
    temperature=0,
    no_speech_threshold=0.3,
    condition_on_previous_text=False,
)

def cleanup_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\.{2,}", ".", text)
    text = text.strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text

def _load_audio(audio_input):
    if isinstance(audio_input, np.ndarray):
        return audio_input
    import soundfile as sf
    audio, _ = sf.read(audio_input, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio

def transcribe_audio(audio_input):
    try:
        audio = _load_audio(audio_input)
        print(f"[ASR] shape={audio.shape}, max={audio.max():.4f}, rms={float(np.sqrt(np.mean(audio**2))):.4f}")
        en_segs, en_info = model.transcribe(audio, task="translate", **_COMMON)
        english_text = cleanup_text(" ".join(s.text.strip() for s in en_segs))
        ml_segs, _ = model.transcribe(audio, task="transcribe", **_COMMON)
        malayalam_text = cleanup_text(" ".join(s.text.strip() for s in ml_segs))
        print(f"English   : {english_text}")
        print(f"Malayalam : {malayalam_text}")
        return {"status": "success", "text": english_text, "malayalam_text": malayalam_text,
                "raw_text": english_text, "language": en_info.language, "segments": [],
                "device": DEVICE, "model": MODEL_NAME}
    except Exception as e:
        print(f"ASR Error: {e}")
        return {"status": "failed", "error": str(e), "text": "", "malayalam_text": "", "segments": []}

def transcribe_audio_stream(audio_input):
    audio = _load_audio(audio_input)
    en_segs, en_info = model.transcribe(audio, task="translate", **_COMMON)
    en_parts = []
    for seg in en_segs:
        t = seg.text.strip()
        if t:
            en_parts.append(t)
            yield {"type": "english_segment", "text": t,
                   "accumulated": cleanup_text(" ".join(en_parts))}
    full_english = cleanup_text(" ".join(en_parts))
    ml_segs, _ = model.transcribe(audio, task="transcribe", **_COMMON)
    ml_parts = []
    for seg in ml_segs:
        t = seg.text.strip()
        if t:
            ml_parts.append(t)
            yield {"type": "malayalam_segment", "text": t,
                   "accumulated": cleanup_text(" ".join(ml_parts))}
    full_malayalam = cleanup_text(" ".join(ml_parts))
    yield {"type": "complete", "english_text": full_english,
           "malayalam_text": full_malayalam, "language": en_info.language}


