from faster_whisper import WhisperModel
import torch
import re
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "small"

print(f"Loading Whisper model on {DEVICE}...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
print("Whisper model loaded successfully")

def cleanup_text(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\.{2,}', '.', text)
    text = text.strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text

def transcribe_audio(audio_input):
    try:
        if isinstance(audio_input, np.ndarray):
            audio = audio_input
        else:
            import soundfile as sf
            audio, sr = sf.read(audio_input, dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

        print(f"Transcribing: shape={audio.shape}, max={audio.max():.3f}")

        # vad_filter=True uses Silero VAD - no ffmpeg/pyannote/torchcodec needed
        segments, info = model.transcribe(
            audio,
            language="ml",
            task="translate",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        text = " ".join([seg.text.strip() for seg in segments])
        cleaned = cleanup_text(text)
        print(f"Transcription: {cleaned}")

        return {
            "status": "success",
            "text": cleaned,
            "raw_text": text,
            "language": info.language,
            "segments": [],
            "device": DEVICE,
            "model": MODEL_NAME
        }
    except Exception as e:
        print(f"ASR Error: {e}")
        return {"status": "failed", "error": str(e), "text": "", "segments": []}
