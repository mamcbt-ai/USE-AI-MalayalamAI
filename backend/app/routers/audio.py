"""
audio.py — Audio processing router
Primary path: raw upload bytes sent directly to Groq (no conversion)
Fallback: WAV conversion only if needed
"""
import json
import os
from datetime import date
from pathlib import Path
import asyncio
import threading

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
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

router        = APIRouter(prefix="/audio", tags=["audio"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
limiter       = Limiter(key_func=get_remote_address)

ALLOWED_LANGS = {"ml", "ta", "te", "kn", "hi"}


# ── Auth ──────────────────────────────────────────────────────────────────────
def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Please login first")
    with Session(engine) as session:
        user = get_user_by_email(session, email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user


def check_usage_limit(user: User) -> None:
    PLAN_LIMITS = {"free": 10, "basic": 30, "pro": 100, "unlimited": 999_999}
    plan  = getattr(user, "plan", "free") or "free"
    limit = PLAN_LIMITS.get(plan, 10)
    with Session(engine) as session:
        all_records = session.exec(select(AudioRecord)).all()
        today_count = sum(
            1 for r in all_records
            if getattr(r, "user_id", None) == user.id
            and r.created_at
            and r.created_at.date() == date.today()
        )
    if today_count >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached ({today_count}/{limit}). Upgrade your plan.",
        )


def _validate_lang(lang: str) -> str:
    lang = (lang or "ml").strip().lower()
    if lang not in ALLOWED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")
    return lang


def _save_record(filename: str, source_lang: str, english_text: str,
                 refined: str, native_text: str) -> None:
    try:
        record = AudioRecord(
            filename=filename,
            language=source_lang,
            transcript=english_text,
            translation=refined,
            malayalam_output=native_text,
        )
        with Session(engine) as session:
            session.add(record)
            session.commit()
    except Exception as db_err:
        print(f"DB save error: {db_err}")


# ── POST /audio/process ───────────────────────────────────────────────────────
@router.post("/process")
@limiter.limit("20/minute")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("standard"),
    lang: str  = Form("ml"),
    current_user: User = Depends(get_current_user),
):
    check_usage_limit(current_user)
    source_lang = _validate_lang(lang)

    # Read raw bytes — pass directly to Groq (no conversion)
    raw_bytes = await file.read()
    filename  = file.filename or "recording.webm"
    suffix    = Path(filename).suffix.lower() or ".webm"

    print(f"audio.py: filename={filename} suffix={suffix} bytes={len(raw_bytes)} content_type={file.content_type}")

    result = transcribe_audio(
        audio_input=raw_bytes,
        filename=filename,
        style=style,
        source_lang=source_lang,
    )

    english_text = result.get("english_text", "")
    native_text  = result.get("native_text",  "")
    lang_name    = result.get("source_language_name", source_lang)
    status       = result.get("status", "failed")

    if status == "too_short":
        return JSONResponse(content={
            "status": "too_short",
            "message": result.get("error", "Recording too short."),
            "english_text": "", "native_text": "",
            "source_lang": source_lang, "source_language_name": lang_name,
        })

    if not english_text and not native_text:
        return JSONResponse(content={
            "status":  "no_speech",
            "message": "No speech detected. Please speak clearly for 5–10 seconds and try again.",
            "english_text": "", "native_text": "",
            "source_lang": source_lang, "source_language_name": lang_name,
        })

    refined = refine_english(english_text, style=style, source_lang=source_lang) if english_text else ""
    _save_record(filename, source_lang, english_text, refined, native_text)

    return JSONResponse(content={
        "status":       "success",
        "english_text": english_text,
        "native_text":  native_text,
        "refined_text": refined,
        "source_lang":  source_lang,
        "source_language_name": lang_name,
        "style":  style,
        "model":  result.get("model", ""),
        "device": result.get("device", ""),
    })


# ── POST /audio/process-stream ────────────────────────────────────────────────
@router.post("/process-stream")
@limiter.limit("20/minute")
async def process_audio_stream(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("standard"),
    lang: str  = Form("ml"),
    current_user: User = Depends(get_current_user),
):
    check_usage_limit(current_user)
    source_lang = _validate_lang(lang)

    raw_bytes = await file.read()
    filename  = file.filename or "recording.webm"

    async def event_generator():
        seg_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            try:
                for seg in transcribe_audio_stream(
                    raw_bytes, filename=filename, style=style, source_lang=source_lang
                ):
                    asyncio.run_coroutine_threadsafe(seg_queue.put(seg), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    seg_queue.put({"type": "error", "error": str(exc)}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(seg_queue.put(None), loop)

        threading.Thread(target=_run, daemon=True).start()

        final_english = ""
        final_native  = ""

        while True:
            seg = await seg_queue.get()
            if seg is None:
                break
            t = seg.get("type", "")
            if t == "error":
                yield f"data: {json.dumps(seg, ensure_ascii=False)}\n\n"
                break
            elif t in ("english_segment", "native_segment", "status"):
                if t == "english_segment":
                    final_english = seg.get("accumulated", "")
                elif t == "native_segment":
                    final_native = seg.get("accumulated", "")
                yield f"data: {json.dumps(seg, ensure_ascii=False)}\n\n"
            elif t == "complete":
                final_english = seg.get("english_text", "")
                final_native  = seg.get("native_text",  "")
                lang_name     = seg.get("source_language_name", source_lang)
                refined       = refine_english(final_english) if final_english else ""
                _save_record(filename, source_lang, final_english, refined, final_native)
                payload = {
                    "type": "complete",
                    "english_text": final_english,
                    "native_text":  final_native,
                    "refined_text": refined,
                    "source_lang":  source_lang,
                    "source_language_name": lang_name,
                    "style": style,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
