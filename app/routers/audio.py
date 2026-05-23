import os
import tempfile
import subprocess
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.services.asr_service import transcribe_audio
from app.services.translation_service import refine_english
from app.services.auth_service import decode_token, get_user_by_email
from app.models.audio_record import AudioRecord
from app.models.user import User
from app.core.db import engine

router = APIRouter(prefix="/audio", tags=["audio"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
limiter = Limiter(key_func=get_remote_address)

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
        from datetime import date
        records_today = session.exec(select(AudioRecord)).all()
        user_records_today = [r for r in records_today if r.created_at and r.created_at.date() == date.today()]
        limit = 10 if user.plan == "free" else 1000
        if len(user_records_today) >= limit:
            raise HTTPException(status_code=429, detail="Daily limit reached. Free plan allows 10 recordings/day.")

@router.post("/process")
@limiter.limit("20/minute")
async def process_audio(request: Request, file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    check_usage_limit(current_user)
    temp_file_path = None
    wav_path = None
    try:
        print(f"Received file: {file.filename}, type: {file.content_type}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            temp_file_path = tmp.name
            content = await file.read()
            tmp.write(content)
        print("Temp audio saved")
        wav_path = temp_file_path.replace(".webm", ".wav")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", temp_file_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, text=True
        )
        print("ffmpeg done:", result.returncode)
        asr_input = wav_path if os.path.exists(wav_path) else temp_file_path
        asr_result = transcribe_audio(asr_input)
        english_text = asr_result.get("text", "").strip()
        if not english_text:
            return {"status": "failed", "message": "No speech detected"}
        print(f"ASR Output: {english_text}")
        refined = refine_english(english_text)
        try:
            record = AudioRecord(filename=file.filename, language=asr_result.get("language", "ml"), transcript=english_text, translation=refined, malayalam_output="")
            with Session(engine) as session:
                session.add(record)
                session.commit()
        except Exception as db_error:
            print(f"Database save error: {db_error}")
        return {"status": "success", "asr_output": asr_result, "english_text": english_text, "refined_text": refined, "malayalam_text": ""}
    except Exception as e:
        print(f"Audio process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [temp_file_path, wav_path]:
            if p and os.path.exists(p):
                os.remove(p)
