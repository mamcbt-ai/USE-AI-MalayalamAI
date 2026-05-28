from faster_whisper import WhisperModel
import torch
import re
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "large-v2"

print(f"Loading Whisper model ({MODEL_NAME}) on {DEVICE}...")
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type="int8")
print(f"Whisper model loaded successfully ({MODEL_NAME}/{DEVICE})")

# ---------------------------------------------------------------------------
# Shared VAD settings
# ---------------------------------------------------------------------------
_VAD = dict(
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 400, "threshold": 0.3},
)

# ---------------------------------------------------------------------------
# Tamil -> Malayalam Unicode corrector
# ---------------------------------------------------------------------------
_TAMIL_TO_ML = {
    '\u0B85': '\u0D05', '\u0B86': '\u0D06', '\u0B87': '\u0D07',
    '\u0B88': '\u0D08', '\u0B89': '\u0D09', '\u0B8A': '\u0D0A',
    '\u0B8E': '\u0D0E', '\u0B8F': '\u0D0F', '\u0B90': '\u0D10',
    '\u0B92': '\u0D12', '\u0B93': '\u0D13', '\u0B94': '\u0D14',
    '\u0B95': '\u0D15', '\u0B99': '\u0D19', '\u0B9A': '\u0D1A',
    '\u0B9C': '\u0D1C', '\u0B9E': '\u0D1E', '\u0B9F': '\u0D1F',
    '\u0BA3': '\u0D23', '\u0BA4': '\u0D24', '\u0BA8': '\u0D28',
    '\u0BA9': '\u0D29', '\u0BAA': '\u0D2A', '\u0BAE': '\u0D2E',
    '\u0BAF': '\u0D2F', '\u0BB0': '\u0D30', '\u0BB1': '\u0D31',
    '\u0BB2': '\u0D32', '\u0BB3': '\u0D33', '\u0BB4': '\u0D34',
    '\u0BB5': '\u0D35', '\u0BB6': '\u0D36', '\u0BB7': '\u0D37',
    '\u0BB8': '\u0D38', '\u0BB9': '\u0D39',
    '\u0BBE': '\u0D3E', '\u0BBF': '\u0D3F', '\u0BC0': '\u0D40',
    '\u0BC1': '\u0D41', '\u0BC2': '\u0D42', '\u0BC6': '\u0D46',
    '\u0BC7': '\u0D47', '\u0BC8': '\u0D48', '\u0BCA': '\u0D4A',
    '\u0BCB': '\u0D4B', '\u0BCC': '\u0D4C', '\u0BCD': '\u0D4D',
    '\u0BE6': '\u0D66', '\u0BE7': '\u0D67', '\u0BE8': '\u0D68',
    '\u0BE9': '\u0D69', '\u0BEA': '\u0D6A', '\u0BEB': '\u0D6B',
    '\u0BEC': '\u0D6C', '\u0BED': '\u0D6D', '\u0BEE': '\u0D6E',
    '\u0BEF': '\u0D6F',
}

_TAMIL_RANGE = range(0x0B80, 0x0C00)


def _fix_ml_script(text: str) -> str:
    """Convert Tamil Unicode codepoints to Malayalam equivalents."""
    if not text:
        return text
    if any(ord(c) in _TAMIL_RANGE for c in text):
        return ''.join(_TAMIL_TO_ML.get(c, c) for c in text)
    return text


# ---------------------------------------------------------------------------
# Dynamic params builder — supports ml, ta, te, kn, hi (and any Whisper lang)
# ---------------------------------------------------------------------------

def _build_params(source_lang: str):
    """
    Build translate + transcribe parameter dicts for the given source language.

    Translate pass  : source_lang -> English
    Transcribe pass : source_lang -> native Unicode
      - For Malayalam (ml) we use language=None to avoid Tamil-token confusion;
        we then apply _fix_ml_script() as a post-processor.
      - For all other languages Whisper handles them natively.
    """
    translate = dict(
        language=source_lang,
        task="translate",
        beam_size=2,
        temperature=[0.0, 0.2],
        no_speech_threshold=0.3,
        condition_on_previous_text=False,
        **_VAD,
    )
    # Malayalam: auto-detect to prevent Tamil-token hallucinations
    transcribe_lang = None if source_lang == "ml" else source_lang
    transcribe = dict(
        language=transcribe_lang,
        task="transcribe",
        beam_size=2,
        temperature=[0.0, 0.2],
        no_speech_threshold=0.3,
        condition_on_previous_text=False,
        **_VAD,
    )
    return translate, transcribe


def _postprocess_script(text: str, source_lang: str) -> str:
    """Apply script-level fixes for the given language."""
    if source_lang == "ml":
        return _fix_ml_script(text)
    # Telugu (te), Kannada (kn), Hindi (hi), Tamil (ta) output correct Unicode natively
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Language display names (used in logs)
# ---------------------------------------------------------------------------
_LANG_NAMES = {
    "ml": "Malayalam", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada",   "hi": "Hindi",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_audio(audio_input, source_lang: str = "ml"):
    try:
        audio = _load_audio(audio_input)
        print(f"[ASR] lang={source_lang} shape={audio.shape}, max={audio.max():.4f}, rms={float(np.sqrt(np.mean(audio**2))):.4f}")

        translate_params, transcribe_params = _build_params(source_lang)

        en_segs, en_info = model.transcribe(audio, **translate_params)
        english_text = cleanup_text(" ".join(s.text.strip() for s in en_segs))

        ml_segs, _ = model.transcribe(audio, **transcribe_params)
        native_raw = " ".join(s.text.strip() for s in ml_segs)
        native_text = cleanup_text(_postprocess_script(native_raw, source_lang))

        lang_name = _LANG_NAMES.get(source_lang, source_lang.upper())
        print(f"English        : {english_text}")
        print(f"{lang_name:<14} : {native_text}")

        return {
            "status": "success",
            "text": english_text,
            "malayalam_text": native_text,
            "raw_text": english_text,
            "language": en_info.language,
            "source_lang": source_lang,
            "segments": [],
            "device": DEVICE,
            "model": MODEL_NAME,
        }
    except Exception as e:
        print(f"ASR Error: {e}")
        return {
            "status": "failed", "error": str(e),
            "text": "", "malayalam_text": "", "segments": [],
        }


def transcribe_audio_stream(audio_input, source_lang: str = "ml"):
    audio = _load_audio(audio_input)
    print(f"[ASR] lang={source_lang} shape={audio.shape}, max={audio.max():.4f}, rms={float(np.sqrt(np.mean(audio**2))):.4f}")

    translate_params, transcribe_params = _build_params(source_lang)

    en_segs, en_info = model.transcribe(audio, **translate_params)
    en_parts = []
    for seg in en_segs:
        t = seg.text.strip()
        if t:
            en_parts.append(t)
            yield {"type": "english_segment", "text": t,
                   "accumulated": cleanup_text(" ".join(en_parts))}

    full_english = cleanup_text(" ".join(en_parts))

    ml_segs, _ = model.transcribe(audio, **transcribe_params)
    ml_parts = []
    for seg in ml_segs:
        t = seg.text.strip()
        if t:
            native_fixed = _postprocess_script(t, source_lang)
            ml_parts.append(native_fixed)
            yield {"type": "malayalam_segment", "text": native_fixed,
                   "accumulated": cleanup_text(" ".join(ml_parts))}

    full_native = cleanup_text(" ".join(ml_parts))

    yield {
        "type": "complete",
        "english_text": full_english,
        "malayalam_text": full_native,
        "language": en_info.language,
        "source_lang": source_lang,
    }