import whisperx
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = whisperx.load_model("large-v3", device=device, language="ml")


def transcribe_audio(file_path: str):
    try:
        audio = whisperx.load_audio(file_path)

        result = model.transcribe(audio, language="ml")

        text = result.get("text", "").strip()

        return {
            "text": text,
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", [])
        }

    except Exception as e:
        return {
            "text": f"Transcription failed: {str(e)}",
            "language": "error",
            "segments": []
        }