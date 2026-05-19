from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlmodel import Session, select
import tempfile

from app.services.asr_service import transcribe_audio
from app.services.translation_service import translate_eng_to_ml, refine_english
from app.services.auth_service import decode_token, get_user_by_email
from app.models.audio_record import AudioRecord
from app.models.user import User
from app.core.db import engine

router = APIRouter(prefix="/audio", tags=["audio"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)):
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Please login to use this feature")
    with Session(engine) as session:
        user = get_user_by_email(session, email)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user


def check_usage_limit(user: User):
    with Session(engine) as session:
        from datetime import datetime, date
        today_start = datetime.combine(date.today(), datetime.min.time())
        records_today = session.exec(
            select(AudioRecord).where(
                AudioRecord.filename != None
            )
        ).all()
        user_records_today = [
            r for r in records_today
            if r.created_at and r.created_at.date() == date.today()
        ]
        limit = 10 if user.plan == "free" else 1000
        if len(user_records_today) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit reached. Free plan: 10 recordings/day. Upgrade for unlimited access."
            )


@router.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    check_usage_limit(current_user)

    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        temp_file_path = tmp.name
        content = await file.read()
        tmp.write(content)

    asr_result = transcribe_audio(temp_file_path)

    english_text = asr_result.get("text", "")
    if not english_text and asr_result.get("segments"):
        english_text = " ".join([seg["text"] for seg in asr_result["segments"]])
    english_text = english_text.strip()

    refined = refine_english(english_text)

    translation_result = {
        "transliteration": english_text,
        "translation": english_text,
        "refined": refined,
        "status": "success"
    }

    reverse_result = translate_eng_to_ml(refined)

    try:
        record = AudioRecord(
            filename=file.filename,
            language=asr_result.get("language", "ml"),
            transcript=english_text,
            translation=english_text,
            malayalam_output=reverse_result.get("malayalam", "")
        )
        with Session(engine) as session:
            session.add(record)
            session.commit()
            print(f"Saved record ID: {record.id}")
    except Exception as e:
        print(f"Database save error: {e}")

    return {
        "asr_output": asr_result,
        "translation_output": translation_result,
        "reverse_translation": reverse_result
    }