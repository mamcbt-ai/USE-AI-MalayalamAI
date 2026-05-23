import whisperx
import torch
import re
import soundfile as sf
import numpy as np

# ========================
# DEVICE & MODEL SETUP
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "small"
print(f"Loading WhisperX model on {DEVICE}...")
model = whisperx.load_model(
    MODEL_NAME,
    device=DEVICE,
    compute_type="int8",
    language="ml"
)
print("WhisperX model loaded successfully")

# ========================
# TEXT CLEANUP
# ========================
def cleanup_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = text.strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text

# ========================
# MAIN TRANSCRIPTION
# ========================
def transcribe_audio(file_path: str):
    try:
        print(f"Processing audio: {file_path}")

        # Load WAV with soundfile (no ffmpeg needed — pure Python)
        audio_data, sample_rate = sf.read(file_path, dtype="float32")
        print(f"Loaded audio: shape={audio_data.shape}, sr={sample_rate}")

        # Convert stereo -> mono if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16000 Hz if not already (WhisperX expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            print(f"Resampled from {sample_rate} to 16000 Hz")

        # Pass numpy array directly — bypasses whisperx.load_audio ffmpeg call
        result = model.transcribe(
            audio_data,
            language="ml",
            task="translate",
            batch_size=16
        )

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
        return {
            "status": "failed",
            "error": str(e),
            "text": "",
            "segments": []
        }
