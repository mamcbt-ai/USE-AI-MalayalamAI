import os
import shutil
import tempfile
import subprocess
import asyncio
import json
import threading
import numpy as np
from datetime import date

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.services.asr_service import transcribe_audio, transcribe_audio_stream
from app.services.translation_service import refine_english
from app.services.auth_service import decode_token, get_user_by_email
from app.models.audio_record import AudioRecord
from app.models.user import User
from app.core.db import engine

router = APIRouter(prefix="/audio", tags=["audio"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
limiter = Limiter(key_func=get_remote_address)

# Detect ffmpeg once at startup
FFMPEG_BIN = shutil.which("ffmpeg")
print(f"audio.py: FFMPEG_BIN={FFMPEG_BIN}")


# ── helpers ──────────────────────────────────────────────────────────────────

def get_current_user(token: str = Depends(oauth2_scheme)):
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Please login first")
    with Session(engine) as session:
        user = get_user_by_email(session, email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user


def check_usage_limit(user: User):
    """Count TODAY's records for THIS user and enforce per-plan daily limits."""
    from datetime import datetime as _dt

    # Check if paid plan has expired → revert to free
    plan = getattr(user, "plan", "free") or "free"
    expires_at = getattr(user, "plan_expires_at", None)
    if plan != "free" and expires_at and _dt.utcnow() > expires_at:
        plan = "free"

    # Per-plan daily limits
    PLAN_LIMITS = {
        "free":      10,
        "basic":     30,
        "pro":       100,
        "unlimited": 999999,
    }
    limit = PLAN_LIMITS.get(plan, 10)

    with Session(engine) as session:
        all_records = session.exec(select(AudioRecord)).all()
        user_today = [
            r for r in all_records
            if getattr(r, "user_id", None) == user.id
            and r.created_at
            and r.created_at.date() == date.today()
        ]
        count = len(user_today)
        if count >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached ({count}/{limit}). Upgrade your plan at /pricing.",
                headers={"X-Plan": plan, "X-Limit": str(limit), "X-Used": str(count)},
            )


async def _decode_audio_to_numpy(file: UploadFile) -> np.ndarray:
    """
    Decode uploaded WebM/Opus (browser MediaRecorder) -> numpy float32 @ 16 kHz.
    Tries ffmpeg first (best quality), falls back to PyAV.
    """
    content = await file.read()

    if FFMPEG_BIN:
        tmp_webm = None
        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
                f.write(content)
                tmp_webm = f.name
            tmp_wav = tmp_webm.replace(".webm", ".wav")
            result = subprocess.run(
                [FFMPEG_BIN, "-y", "-i", tmp_webm,
                 "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and os.path.exists(tmp_wav):
                import soundfile as sf
                audio, _ = sf.read(tmp_wav, dtype="float32")
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                print(f"ffmpeg decode OK: {audio.shape}")
                return audio
            else:
                print(f"ffmpeg error: {result.stderr[:200]}")
        except Exception as e:
            print(f"ffmpeg decode exception: {e}")
        finally:
            for p in [tmp_webm, tmp_wav]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    # PyAV fallback (no ffmpeg binary needed)
    try:
        import av
        import io as _io
        container = av.open(_io.BytesIO(content))
        samples = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            samples.append(arr.astype(np.float32))
        if samples:
            audio = np.concatenate(samples)
            if audio.max() > 1.5:
                audio = audio / 32768.0
            print(f"PyAV decode OK: {audio.shape}, max={audio.max():.3f}")
            return audio
    except Exception as e:
        print(f"PyAV decode error: {e}")

    raise HTTPException(status_code=400, detail="Could not decode audio. Please try again.")


def _save_record(filename, source_lang, english_text, refined, native_text):
    try:
        record = AudioRecord(
            filename=filename,
            language=source_lang,
            transcript=english_text,
            translation=refined,
            malayalam_output=native_text,  # kept for DB compat
        )
        with Session(engine) as session:
            session.add(record)
            session.commit()
    except Exception as db_err:
        print(f"DB save error: {db_err}")


# ── POST /audio/process  (JSON response, primary endpoint) ───────────────────

@router.post("/process")
@limiter.limit("20/minute")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("standard"),
    lang: str = Form("ml"),
    current_user: User = Depends(get_current_user),
):
    check_usage_limit(current_user)

    audio = await _decode_audio_to_numpy(file)
    asr_result = transcribe_audio(audio, style=style, source_lang=lang)

    english_text = asr_result.get("english_text", "").strip()
    native_text  = asr_result.get("native_text", "").strip()
    source_lang  = asr_result.get("source_lang", lang)
    lang_name    = asr_result.get("source_language_name", lang)

    if not english_text and not native_text:
        return {
            "status": "failed",
            "message": "No speech detected. Please speak clearly and try again.",
            "english_text": "",
            "native_text": "",
            "source_lang": source_lang,
            "source_language_name": lang_name,
        }

    refined = refine_english(english_text)
    _save_record(file.filename, source_lang, english_text, refined, native_text)

    return {
        "status": "success",
        "english_text": english_text,
        "native_text": native_text,
        "refined_text": refined,
        "source_lang": source_lang,
        "source_language_name": lang_name,
        "model": asr_result.get("model", ""),
        "device": asr_result.get("device", ""),
    }


# ── POST /audio/process-stream  (SSE live streaming) ─────────────────────────

@router.post("/process-stream")
@limiter.limit("20/minute")
async def process_audio_stream(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("standard"),
    lang: str = Form("ml"),
    current_user: User = Depends(get_current_user),
):
    """
    SSE streaming endpoint. Events:
      {"type":"status",          "message":"..."}
      {"type":"english_segment", "text":"...", "accumulated":"..."}
      {"type":"native_segment",  "text":"...", "accumulated":"..."}
      {"type":"complete",        "english_text":"...", "native_text":"...", "refined_text":"...", "source_lang":"...", "source_language_name":"..."}
      {"type":"done"}
    """
    check_usage_limit(current_user)

    audio = await _decode_audio_to_numpy(file)
    filename = file.filename or "recording.webm"

    async def event_generator():
        yield f"data: {json.dumps({'type': 'status', 'message': 'Audio received - starting transcription...'})}\n\n"

        seg_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run_asr():
            try:
                for seg in transcribe_audio_stream(audio, style=style, source_lang=lang):
                    asyncio.run_coroutine_threadsafe(seg_queue.put(seg), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    seg_queue.put({"type": "error", "message": str(exc)}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(seg_queue.put(None), loop)

        threading.Thread(target=_run_asr, daemon=True).start()

        final_english = ""
        final_native  = ""
        final_lang    = lang

        while True:
            seg = await seg_queue.get()
            if seg is None:
                break

            t = seg.get("type", "")

            if t == "error":
                yield f"data: {json.dumps(seg)}\n\n"
                return

            elif t in ("english_segment", "native_segment"):
                if t == "english_segment":
                    final_english = seg.get("accumulated", "")
                else:
                    final_native = seg.get("accumulated", "")
                yield f"data: {json.dumps(seg)}\n\n"

            elif t == "complete":
                final_english = seg.get("english_text", "")
                final_native  = seg.get("native_text", "")
                final_lang    = seg.get("source_lang", lang)
                lang_name     = seg.get("source_language_name", lang)

                refined = refine_english(final_english)
                _save_record(filename, final_lang, final_english, refined, final_native)

                yield f"data: {json.dumps({'type': 'complete', 'english_text': final_english, 'native_text': final_native, 'refined_text': refined, 'source_lang': final_lang, 'source_language_name': lang_name})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
