import whisperx
import torch
import re
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "small"
print(f"Loading WhisperX model on {DEVICE}...")
model = whisperx.load_model(MODEL_NAME, device=DEVICE, compute_type="int8", language="ml")
print("WhisperX model loaded successfully")

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
        # Accept either numpy array or file path
        if isinstance(audio_input, np.ndarray):
            audio = audio_input
            print(f"Transcribing numpy array: shape={audio.shape}, max={audio.max():.3f}")
        else:
            import soundfile as sf
            audio, sr = sf.read(audio_input, dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            print(f"Transcribing file: {audio_input}, shape={audio.shape}")

        result = model.transcribe(audio, language="ml", task="translate", batch_size=16)
        raw_text = result.get("text", "").strip()
        cleaned_text = cleanup_text(raw_text)
        print(f"Transcription result: {cleaned_text}")
        return {
            "status": "success",
            "text": cleaned_text,
            "raw_text": raw_text,
            "language": result.get("language", "ml"),
            "segments": result.get("segments", []),
            "device": DEVICE,
            "model": MODEL_NAME
        }
    except Exception as e:
        print(f"ASR Error: {e}")
        return {"status": "failed", "error": str(e), "text": "", "segments": []}
