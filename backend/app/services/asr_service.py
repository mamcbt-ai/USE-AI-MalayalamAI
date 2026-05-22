import whisperx
import torch
import re

# ========================
# DEVICE & MODEL SETUP
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Faster model
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

        # Load audio
        audio = whisperx.load_audio(file_path)

        # Faster transcription
        result = model.transcribe(
    audio,
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