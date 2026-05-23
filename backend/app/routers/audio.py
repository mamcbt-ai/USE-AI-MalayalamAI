# v7 - glob nix store to find real ffmpeg binary for WebM conversion
import os
import glob
import shutil
import tempfile
import subprocess
import numpy as np
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

def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    nix = glob.glob("/nix/store/*/bin/ffmpeg")
    if nix:
        print(f"v7: found ffmpeg at {nix[0]}")
        return nix[0]
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(p):
            return p
    return None

FFMPEG_BIN = find_ffmpeg()
print(f"v7: FFMPEG_BIN={FFMPEG_BIN}")

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
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            temp_file_path = tmp.name
            content = await file.read()
            tmp.write(content)
        print(f"v7: {len(content)} bytes, FFMPEG_BIN={FFMPEG_BIN}")

        wav_path = temp_file_path.replace(".webm", ".wav")

        if FFMPEG_BIN:
            result = subprocess.run(
                [FFMPEG_BIN, "-y", "-i", temp_file_path,
                 "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, text=True, timeout=30
            )
            print(f"v7: ffmpeg rc={result.returncode}")
            if result.returncode != 0:
                print(f"v7: ffmpeg stderr={result.stderr[:200]}")

            if os.path.exists(wav_path) and result.returncode == 0:
                import soundfile as sf
                audio_data, sr = sf.read(wav_path, dtype="float32")
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                print(f"v7: wav loaded shape={audio_data.shape}, max={audio_data.max():.4f}")
                asr_input = audio_data
            else:
                asr_input = temp_file_path
        else:
            # fallback: PyAV numpy decode
            import av
            container = av.open(temp_file_path)
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            frames = []
            try:
                for frame in container.decode(audio=0):
                    try:
                        for rf in resampler.resample(frame):
                            frames.append(rf.to_ndarray())
                    except Exception:
                        continue
            except Exception as e:
                print(f"v7: PyAV warning: {e}")
            finally:
                try:
                    for rf in resampler.resample(None):
                        frames.append(rf.to_ndarray())
                except Exception:
                    pass
                container.close()
            if frames:
                audio_data = np.concatenate(frames, axis=1).flatten().astype(np.float32) / 32768.0
            else:
                audio_data = np.zeros(16000, dtype=np.float32)
            print(f"v7: PyAV fallback shape={audio_data.shape}, max={audio_data.max():.4f}")
            asr_input = audio_data

        asr_result = transcribe_audio(asr_input)
        english_text = asr_result.get("text", "").strip()
        print(f"v7: ASR={english_text[:80]}")

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
        print(f"v7 exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [temp_file_path, wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
