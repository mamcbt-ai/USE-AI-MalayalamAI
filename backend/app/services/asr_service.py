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
    # Short prompts — long prompts get echoed back by Whisper on silent/short audio
    prompts = {
        "formal":         "Formal English translation.",
        "casual":         "Casual English translation.",
        "official":       "Official English translation.",
        "professional":   "Professional English translation.",
        "friendly":       "Friendly English translation.",
        "conversational": "Conversational English translation.",
        "social_media":   "Social media style English.",
        "business":       "Business English translation.",
        "emotional":      "Emotional English translation.",
        "cinematic":      "Cinematic English translation.",
        "academic":       "Academic English translation.",
        "news":           "News style English translation.",
        "literary":       "Literary English translation.",
        "simple":         "Simple English translation.",
        "humorous":       "Humorous English translation.",
        "bullet_points":  "English bullet points.",
        "standard":       "English translation.",
    }
    return prompts.get(style, "English translation.")


def _is_prompt_echo(text: str, style: str) -> bool:
    """Return True if Whisper just echoed the prompt instead of transcribing."""
    if not text:
        return False
    lower = text.lower().strip(".")
    echo_phrases = [
        "english translation", "formal english", "casual english",
        "official english", "professional english", "friendly english",
        "conversational english", "social media style", "business english",
        "emotional english", "cinematic english", "academic english",
        "news style english", "literary english", "simple english",
        "humorous english", "english bullet points",
        "translate this malayalam",
    ]
    return any(lower.startswith(p) for p in echo_phrases)


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
        if _is_prompt_echo(english_text, style):
            print(f"Prompt echo detected, clearing english_text: {english_text}")
            english_text = ""

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
        if t and not _is_prompt_echo(t, style):
            en_parts.append(t)
            yield {
                "type": "english_segment",
                "text": t,
                "accumulated": cleanup_text(" ".join(en_parts)),
            }
    full_english = cleanup_text(" ".join(en_parts))
    if _is_prompt_echo(full_english, style):
        full_english = ""

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
