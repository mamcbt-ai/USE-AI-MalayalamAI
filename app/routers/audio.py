"""
audio.py
Malayalam Voice AI — Audio processing endpoints

POST /audio/process         — standard (blocking) transcription + translation
POST /audio/process-stream  — streaming SSE transcription + translation

Both endpoints now accept an optional `style` field in the multipart form.
The style value is passed to translation_service.refine_translation().
Valid style values: standard, formal, casual, news, literary, business,
                    academic, simple, humorous, emotional, bullet
If omitted or unrecognised, defaults to 'standard'.
"""

import asyncio
import threading
import logging
from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from auth import get_current_user
from database import get_session
from models import User, AudioRecord
from asr_service import transcribe_audio
from translation_service import refine_translation, available_styles, DEFAULT_STYLE
from usage import check_usage_limit, record_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audio", tags=["audio"])


# ---------------------------------------------------------------------------
# POST /audio/process  — blocking endpoint
# ---------------------------------------------------------------------------

@router.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE),          # ← new: style chip from frontend
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """
    Transcribe uploaded audio and return English translation + Malayalam text.

    Form fields:
        file   — audio file (webm/opus, mp3, wav)
        style  — translation style key (default: 'standard')
    """
    # --- Usage gate ---
    if not check_usage_limit(db, current_user):
        raise HTTPException(
            status_code=429,
            detail="Daily transcription limit reached. Please upgrade your plan.",
        )

    # --- Read audio bytes ---
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- ASR: dual-pass Whisper (translate → English, transcribe → Malayalam) ---
    asr_result = transcribe_audio(audio_bytes)
    raw_english: str = asr_result.get("translation_text", "")
    malayalam_text: str = asr_result.get("malayalam_text", "")

    # --- Refinement: GPT rewrite with requested style ---
    resolved_style = style.lower().strip() if style else DEFAULT_STYLE
    refined_english = refine_translation(raw_english, style=resolved_style)

    # --- Persist usage record ---
    record_usage(db, current_user, file.filename or "upload")

    return {
        "success": True,
        "style": resolved_style,
        "raw_text": raw_english,
        "refined_text": refined_english,       # frontend reads this field
        "malayalam_text": malayalam_text,
        "available_styles": available_styles(), # handy for frontend validation
    }


# ---------------------------------------------------------------------------
# POST /audio/process-stream  — SSE streaming endpoint
# ---------------------------------------------------------------------------

@router.post("/process-stream")
async def process_audio_stream(
    file: UploadFile = File(...),
    style: str = Form(DEFAULT_STYLE),          # ← new: style chip from frontend
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """
    Transcribe uploaded audio and stream results as Server-Sent Events.

    SSE event types emitted:
        segment   — incremental English segment from Whisper
        refined   — full GPT-refined English text (sent at end)
        malayalam — full Malayalam transcription (sent at end)
        done      — signals stream completion
        error     — signals a failure
    """
    if not check_usage_limit(db, current_user):
        raise HTTPException(
            status_code=429,
            detail="Daily transcription limit reached. Please upgrade your plan.",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    resolved_style = style.lower().strip() if style else DEFAULT_STYLE

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def run_transcription():
            """Runs in a background thread; pushes SSE events onto the queue."""
            try:
                result = transcribe_audio(audio_bytes, segment_callback=lambda seg: (
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        f"event: segment\ndata: {seg}\n\n"
                    )
                ))

                raw_english = result.get("translation_text", "")
                malayalam_text = result.get("malayalam_text", "")

                # Refine with style
                refined_english = refine_translation(raw_english, style=resolved_style)

                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    f"event: refined\ndata: {refined_english}\n\n"
                )
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    f"event: malayalam\ndata: {malayalam_text}\n\n"
                )
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    f"event: done\ndata: style={resolved_style}\n\n"
                )

            except Exception as exc:
                logger.error("Streaming transcription error: %s", exc)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    f"event: error\ndata: {str(exc)}\n\n"
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = threading.Thread(target=run_transcription, daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        record_usage(db, current_user, file.filename or "upload")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # disables Nginx response buffering
        },
    )
