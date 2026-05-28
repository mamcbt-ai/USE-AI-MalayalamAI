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

FFMPEG_BIN = shutil.which("ffmpeg")
print(f"audio.py: FFMPEG_BIN={FFMPEG_BIN}")

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
    with Session(engine) as session:
        all_records = session.exec(select(AudioRecord)).all()
        user_today = [r for r in all_records
            if getattr(r, "user_id", None) == user.id
            and r.created_at and r.created_at.date() == date.today()]
        limit = 10 if user.plan == "free" else 1000
        if len(user_today) >= limit:
            raise HTTPException(status_code=429, detail="Daily limit reached.")

async def _decode_audio_to_numpy(file: UploadFile):
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
                [FFMPEG_BIN, "-y", "-i", tmp_webm, "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav],
                capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and os.path.exists(tmp_wav):
                import soundfile as sf
                audio, _ = sf.read(tmp_wav, dtype="float32")
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                return audio
        except Exception as e:
            print(f"ffmpeg error: {e}")
        finally:
            for p in [tmp_webm, tmp_wav]:
                if p and os.path.exists(p):
                    try: os.remove(p)
                    except: pass
    try:
        import av, io as _io
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
            return audio
    except Exception as e:
        print(f"PyAV error: {e}")
    raise HTTPException(status_code=400, detail="Could not decode audio.")

def _save_record(filename, language, english_text, refined, malayalam_text):
    try:
        record = AudioRecord(filename=filename, language=language,
            transcript=english_text, translation=refined, malayalam_output=malayalam_text)
        with Session(engine) as session:
            session.add(record)
            session.commit()
    except Exception as e:
        print(f"DB save error: {e}")

@router.post("/process")
@limiter.limit("20/minute")
async def process_audio(request: Request, file: UploadFile = File(...), style: str = Form("standard"),
                         current_user: User = Depends(get_current_user)):
    check_usage_limit(current_user)
    audio = await _decode_audio_to_numpy(file)
    asr_result = transcribe_audio(audio)
    english_text = asr_result.get("text", "").strip()
    malayalam_text = asr_result.get("malayalam_text", "").strip()
    if not english_text:
        return {"status": "failed", "message": "No speech detected"}
    refined = refine_english(english_text)
    _save_record(file.filename, asr_result.get("language", "ml"), english_text, refined, malayalam_text)
    return {"status": "success", "asr_output": asr_result, "english_text": english_text,
            "refined_text": refined, "malayalam_text": malayalam_text}

@router.post("/process-stream")
@limiter.limit("20/minute")
async def process_audio_stream(request: Request, file: UploadFile = File(...), style: str = Form("standard"),
                                current_user: User = Depends(get_current_user)):
    check_usage_limit(current_user)
    audio = await _decode_audio_to_numpy(file)
    filename = file.filename or "recording.webm"

    async def event_generator():
        yield f"data: {json.dumps({'type': 'status', 'message': 'Transcription starting...'})}\n\n"
        seg_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        def _run():
            try:
                for seg in transcribe_audio_stream(audio):
                    asyncio.run_coroutine_threadsafe(seg_queue.put(seg), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    seg_queue.put({"type": "error", "message": str(exc)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(seg_queue.put(None), loop)
        threading.Thread(target=_run, daemon=True).start()
        final_english = ""
        final_malayalam = ""
        while True:
            seg = await seg_queue.get()
            if seg is None:
                break
            t = seg.get("type", "")
            if t == "error":
                yield f"data: {json.dumps(seg)}\n\n"
                return
            elif t in ("english_segment", "malayalam_segment"):
                if t == "english_segment":
                    final_english = seg.get("accumulated", "")
                else:
                    final_malayalam = seg.get("accumulated", "")
                yield f"data: {json.dumps(seg)}\n\n"
            elif t == "complete":
                final_english = seg.get("english_text", "")
                final_malayalam = seg.get("malayalam_text", "")
                lang = seg.get("language", "ml")
                refined = refine_english(final_english, style=style)
                _save_record(filename, lang, final_english, refined, final_malayalam)
                yield f"data: {json.dumps({'type': 'complete', 'english_text': final_english, 'refined_text': refined, 'malayalam_text': final_malayalam})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
