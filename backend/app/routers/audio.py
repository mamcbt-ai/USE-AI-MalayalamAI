import os, tempfile
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select
import json, threading, asyncio
from datetime import date
from app.services.asr_service import transcribe_audio, transcribe_audio_stream
from app.services.translation_service import refine_english
from app.services.auth_service import decode_token, get_user_by_email
from app.models.audio_record import AudioRecord
from app.models.user import User
from app.core.db import engine
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter(prefix="/audio", tags=["audio"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
limiter = Limiter(key_func=get_remote_address)

PLAN_LIMITS = {"free":10,"basic":50,"pro":200,"unlimited":999999,"premium":999999}

def get_current_user(token: str = Depends(oauth2_scheme)):
    email = decode_token(token)
    if not email: raise HTTPException(status_code=401, detail="Please login first")
    with Session(engine) as session:
        user = get_user_by_email(session, email)
        if not user: raise HTTPException(status_code=401, detail="User not found")
        return user

def check_usage_limit(user: User):
    from datetime import datetime as _dt
    plan = getattr(user,"plan","free") or "free"
    expires = getattr(user,"plan_expires_at",None)
    if plan != "free" and expires and _dt.utcnow() > expires: plan = "free"
    limit = PLAN_LIMITS.get(plan, 10)
    with Session(engine) as session:
        records = session.exec(select(AudioRecord)).all()
        today_count = sum(1 for r in records if getattr(r,"user_id",None)==user.id and r.created_at and r.created_at.date()==date.today())
        if today_count >= limit:
            raise HTTPException(status_code=429, detail=f"Daily limit reached ({today_count}/{limit}). Upgrade at /pricing.")

def _save_record(filename, language, english_text, refined, native_text):
    try:
        rec = AudioRecord(filename=filename, language=language, transcript=english_text, translation=refined, malayalam_output=native_text)
        with Session(engine) as session:
            session.add(rec); session.commit()
    except Exception as e:
        print(f"DB save error: {e}")

@router.post("/process")
@limiter.limit("20/minute")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("standard"),
    lang:  str = Form("ml"),
    current_user: User = Depends(get_current_user),
):
    check_usage_limit(current_user)
    raw_bytes = await file.read()
    asr = transcribe_audio(raw_bytes, style=style, source_lang=lang)
    eng     = asr.get("english_text") or asr.get("text","")
    native  = asr.get("native_text")  or asr.get("malayalam_text","")
    if not eng and not native:
        return {"status":"failed","message":"No speech detected"}
    refined = refine_english(eng)
    _save_record(file.filename, asr.get("language",lang), eng, refined, native)
    return {
        "status":       "success",
        "english_text": eng,
        "native_text":  native,
        "malayalam_text": native,
        "refined_text": refined,
        "text":         eng,
        "language":     asr.get("language", lang),
        "language_name":asr.get("language_name", lang),
        "asr_output":   asr,
    }

@router.post("/process-stream")
@limiter.limit("20/minute")
async def process_audio_stream(
    request: Request,
    file: UploadFile = File(...),
    style: str = Form("standard"),
    lang:  str = Form("ml"),
    current_user: User = Depends(get_current_user),
):
    check_usage_limit(current_user)
    raw_bytes = await file.read()
    filename  = file.filename or "recording.webm"

    async def event_generator():
        yield f"data: {json.dumps({'type':'status','message':'Audio received — processing...'})}\n\n"
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _run():
            try:
                for seg in transcribe_audio_stream(raw_bytes, style=style, source_lang=lang):
                    asyncio.run_coroutine_threadsafe(queue.put(seg), loop)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(queue.put({"type":"error","message":str(exc)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_run, daemon=True).start()
        final_eng, final_native = "", ""
        while True:
            seg = await queue.get()
            if seg is None: break
            t = seg.get("type","")
            if t == "error":
                yield f"data: {json.dumps(seg)}\n\n"; return
            elif t in ("english_segment","malayalam_segment","native_segment"):
                if t == "english_segment": final_eng    = seg.get("accumulated","")
                else:                       final_native = seg.get("accumulated","")
                yield f"data: {json.dumps(seg)}\n\n"
            elif t == "complete":
                final_eng    = seg.get("english_text","")
                final_native = seg.get("native_text","") or seg.get("malayalam_text","")
                refined = refine_english(final_eng)
                _save_record(filename, lang, final_eng, refined, final_native)
                yield f"data: {json.dumps({'type':'complete','english_text':final_eng,'native_text':final_native,'malayalam_text':final_native,'refined_text':refined})}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"})
