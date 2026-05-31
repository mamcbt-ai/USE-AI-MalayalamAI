from faster_whisper import WhisperModel
import torch
import re
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "small"

print(f"Loading Whisper model ({MODEL_NAME}) on {DEVICE}...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
print(f"Whisper model loaded successfully ({MODEL_NAME}/{DEVICE})")

_BASE_COMMON = dict(
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

def _style_prompt(style: str) -> str:
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

def _is_hallucination(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    from collections import Counter
    counts = Counter(text.replace(" ", ""))
    if counts:
        most_common_ratio = counts.most_common(1)[0][1] / len(text.replace(" ", ""))
        if most_common_ratio > 0.6:
            print(f"Hallucination detected (repeated chars): {text[:50]}")
            return True
    hallucination_phrases = [
        "thank you", "thanks for watching", "subscribe", "music",
        "♪", "[ music ]", "[music]", "subtitles", "captions",
    ]
    lower = text.lower()
    if any(p in lower for p in hallucination_phrases) and len(text) < 30:
        print(f"Hallucination detected (stock phrase): {text[:50]}")
        return True
    return False

def transcribe_audio(audio_input, style: str = "standard", source_lang: str = "ml") -> dict:
    try:
        audio = _load_audio(audio_input)
        print(f"Transcribing: shape={audio.shape}, max={audio.max():.3f}, lang={source_lang}")

        common = {**_BASE_COMMON, "language": source_lang}

        en_segs, en_info = model.transcribe(
            audio,
            task="translate",
            initial_prompt=_style_prompt(style),
            **common,
        )
        english_text = cleanup_text(" ".join(s.text.strip() for s in en_segs))
        if _is_prompt_echo(english_text, style):
            english_text = ""

        ml_segs, _ = model.transcribe(
            audio,
            task="transcribe",
            **common,
        )
        native_text = cleanup_text(" ".join(s.text.strip() for s in ml_segs))
        if _is_hallucination(native_text):
            native_text = ""

        print(f"English  : {english_text}")
        print(f"Native   : {native_text}")

        return {
            "status": "success",
            "text": english_text,
            "malayalam_text": native_text,
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

def transcribe_audio_stream(audio_input, style: str = "standard", source_lang: str = "ml"):
    audio = _load_audio(audio_input)
    common = {**_BASE_COMMON, "language": source_lang}

    en_segs, en_info = model.transcribe(
        audio,
        task="translate",
        initial_prompt=_style_prompt(style),
        **common,
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

    ml_segs, _ = model.transcribe(
        audio,
        task="transcribe",
        **common,
    )
    ml_parts = []
    for seg in ml_segs:
        t = seg.text.strip()
        if t and not _is_hallucination(t):
            ml_parts.append(t)
            yield {
                "type": "malayalam_segment",
                "text": t,
                "accumulated": cleanup_text(" ".join(ml_parts)),
            }
    full_native = cleanup_text(" ".join(ml_parts))
    if _is_hallucination(full_native):
        full_native = ""

    yield {
        "type": "complete",
        "english_text": full_english,
        "malayalam_text": full_native,
        "language": en_info.language,
    }