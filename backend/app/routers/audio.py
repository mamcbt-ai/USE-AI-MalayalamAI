# v6 - PyAV direct numpy load, no WAV file, no system ffmpeg
import os
import av
import numpy as np
import tempfile
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


def webm_to_numpy(input_path: str, target_sr: int = 16000) -> np.ndarray:
    """Decode WebM/Opus directly to float32 numpy array using PyAV. Skips corrupt packets."""
    container = av.open(input_path)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    frames = []
    try:
        for frame in container.decode(audio=0):
            try:
                for rf in resampler.resample(frame):
                    frames.append(rf.to_ndarray())
            except Exception:
                continue  # skip corrupt Opus packets
    except Exception as e:
        print(f"PyAV decode warning: {e}")
    finally:
        try:
            for rf in resampler.resample(None):
                frames.append(rf.to_ndarray())
        except Exception:
            pass
        container.close()

    if not frames:
        return np.zeros(target_sr, dtype=np.float32)

    audio = np.concatenate(frames, axis=1).flatten().astype(np.float32) / 32768.0
    print(f"v6: decoded {len(audio)} samples, max_amp={audio.max():.4f}")
    return audio


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
            raise HTTPException(status_code=429, detail="Daily limit reached.")


@router.post("/process")
@limiter.limit("20/minute")
async def process_audio(request: Request, file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    check_usage_limit(current_user)
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            temp_file_path = tmp.name
            content = await file.read()
            tmp.write(content)
        print(f"v6: received {len(content)} bytes")

        audio_array = webm_to_numpy(temp_file_path)

        if audio_array.max() < 0.001:
            print("v6: audio appears silent")
            return {"status": "failed", "message": "No speech detected (silent audio)"}

        asr_result = transcribe_audio(audio_array)
        english_text = asr_result.get("text", "").strip()

        if not english_text:
            return {"status": "failed", "message": "No speech detected"}

        refined = refine_english(english_text)

        try:
            record = AudioRecord(
                filename=file.filename,
                language=asr_result.get("language", "ml"),
                transcript=english_text,
                translation=refined,
                malayalam_output=""
            )
            with Session(engine) as session:
                session.add(record)
                session.commit()
        except Exception as db_error:
            print(f"DB save error: {db_error}")

        return {
            "status": "success",
            "asr_output": asr_result,
            "english_text": english_text,
            "refined_text": refined,
            "malayalam_text": ""
        }

    except Exception as e:
        print(f"v6 exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
