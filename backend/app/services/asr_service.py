from faster_whisper import WhisperModel
import torch
import re
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "small"   # small is 3x faster than medium on CPU with good Malayalam accuracy

print(f"Loading Whisper model ({MODEL_NAME}) on {DEVICE}...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
print(f"Whisper model loaded successfully ({MODEL_NAME}/{DEVICE})")

# ── shared Whisper kwargs ────────────────────────────────────────────────────
_COMMON = dict(
    language="ml",
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 300},
    beam_size=5,
    best_of=5,
    temperature=0,
    condition_on_previous_text=True,
)


def cleanup_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\.{2,}", ".", text)
    text = text.strip()
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    return text


def _load_audio(audio_input) -> np.ndarray:
    if isinstance(audio_input, np.ndarray):
        return audio_input
    import soundfile as sf
    audio, _ = sf.read(audio_input, dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio


# ── non-streaming (used by /audio/process) ──────────────────────────────────

def _style_prompt(style: str) -> str:
    prompts = {
        "formal":         "Translate this Malayalam speech into formal English:",
        "casual":         "Translate this Malayalam speech into casual conversational English:",
        "official":       "Translate this Malayalam speech into official English:",
        "professional":   "Translate this Malayalam speech into professional English:",
        "friendly":       "Translate this Malayalam speech into friendly English:",
        "conversational": "Translate this Malayalam speech into natural conversational English:",
        "social_media":   "Translate this Malayalam speech into social media style English:",
        "business":       "Translate this Malayalam speech into business English:",
        "emotional":      "Translate this Malayalam speech capturing emotions in English:",
        "cinematic":      "Translate this Malayalam speech into cinematic narrative English:",
        "academic":       "Translate this Malayalam speech into academic English:",
        "news":           "Translate this Malayalam speech into news broadcast English:",
        "literary":       "Translate this Malayalam speech into literary English:",
        "simple":         "Translate this Malayalam speech into simple plain English:",
        "humorous":       "Translate this Malayalam speech into humorous English:",
        "bullet_points":  "Translate this Malayalam speech into bullet point English:",
        "standard":       "Translate this Malayalam speech accurately into English:",
    }
    return prompts.get(style, "Translate this Malayalam speech accurately into English:")


def transcribe_audio(audio_input, style: str = "standard") -> dict:
    """Transcribe audio; returns both English translation and Malayalam Unicode."""
    try:
        audio = _load_audio(audio_input)
        print(f"Transcribing: shape={audio.shape}, max={audio.max():.3f}")

        # Pass 1 – English translation
        en_segs, en_info = model.transcribe(
            audio,
            task="translate",
            initial_prompt=_style_prompt(style),
            **_COMMON,
        )
        english_text = cleanup_text(" ".join(s.text.strip() for s in en_segs))

        # Pass 2 – Malayalam Unicode
        ml_segs, _ = model.transcribe(
            audio,
            task="transcribe",
            **_COMMON,
        )
        malayalam_text = cleanup_text(" ".join(s.text.strip() for s in ml_segs))

        print(f"English   : {english_text}")
        print(f"Malayalam : {malayalam_text}")

        return {
            "status": "success",
            "text": english_text,
            "malayalam_text": malayalam_text,
            "raw_text": english_text,
            "language": en_info.language,
            "segments": [],
            "device": DEVICE,
            "model": MODEL_NAME,
        }

    except Exception as e:
        print(f"ASR Error: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "text": "",
            "malayalam_text": "",
            "segments": [],
        }


# ── streaming generator (used by /audio/process-stream) ─────────────────────

def transcribe_audio_stream(audio_input, style: str = "standard"):
    """
    Yields dicts as Whisper decodes segments.

    Yielded types:
      {"type": "english_segment",  "text": "...", "accumulated": "..."}
      {"type": "malayalam_segment","text": "...", "accumulated": "..."}
      {"type": "complete", "english_text": "...", "malayalam_text": "...", "language": "..."}
    """
    audio = _load_audio(audio_input)

    # Pass 1 – English translation (stream segments live)
    en_segs, en_info = model.transcribe(
        audio,
        task="translate",
        initial_prompt=_style_prompt(style),
        **_COMMON,
    )
    en_parts = []
    for seg in en_segs:
        t = seg.text.strip()
        if t:
            en_parts.append(t)
            yield {
                "type": "english_segment",
                "text": t,
                "accumulated": cleanup_text(" ".join(en_parts)),
            }
    full_english = cleanup_text(" ".join(en_parts))

    # Pass 2 – Malayalam Unicode (stream segments live)
    ml_segs, _ = model.transcribe(
        audio,
        task="transcribe",
        **_COMMON,
    )
    ml_parts = []
    for seg in ml_segs:
        t = seg.text.strip()
        if t:
            ml_parts.append(t)
            yield {
                "type": "malayalam_segment",
                "text": t,
                "accumulated": cleanup_text(" ".join(ml_parts)),
            }
    full_malayalam = cleanup_text(" ".join(ml_parts))

    yield {
        "type": "complete",
        "english_text": full_english,
        "malayalam_text": full_malayalam,
        "language": en_info.language,
    }
