import os
import tempfile

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException,
    Request
)

from fastapi.security import OAuth2PasswordBearer

from sqlmodel import Session, select

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.services.asr_service import transcribe_audio

from app.services.translation_service import (
    refine_english
)

from app.services.auth_service import (
    decode_token,
    get_user_by_email
)

from app.models.audio_record import AudioRecord
from app.models.user import User

from app.core.db import engine

# =========================
# Router
# =========================
router = APIRouter(
    prefix="/audio",
    tags=["audio"]
)

# =========================
# OAuth
# =========================
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)

# =========================
# Rate Limiter
# =========================
limiter = Limiter(
    key_func=get_remote_address
)

# =========================
# Get Current User
# =========================
def get_current_user(
    token: str = Depends(oauth2_scheme)
):

    email = decode_token(token)

    if not email:
        raise HTTPException(
            status_code=401,
            detail="Please login first"
        )

    with Session(engine) as session:

        user = get_user_by_email(
            session,
            email
        )

        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )

        return user

# =========================
# Daily Usage Limit
# =========================
def check_usage_limit(user: User):

    with Session(engine) as session:

        from datetime import date

        records_today = session.exec(
            select(AudioRecord)
        ).all()

        user_records_today = [
            r for r in records_today
            if r.created_at
            and r.created_at.date() == date.today()
        ]

        # Free vs Paid
        limit = 10 if user.plan == "free" else 1000

        if len(user_records_today) >= limit:

            raise HTTPException(
                status_code=429,
                detail=(
                    "Daily limit reached. "
                    "Free plan allows 10 recordings/day."
                )
            )

# =========================
# Audio Processing Endpoint
# =========================
@router.post("/process")
@limiter.limit("20/minute")
async def process_audio(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):

    # =========================
    # Check Plan Limit
    # =========================
    check_usage_limit(current_user)

    temp_file_path = None

    try:

        print(f"Received file: {file.filename}")

        # =========================
        # Save Temp Audio File
        # =========================
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            temp_file_path = tmp.name

            content = await file.read()

            tmp.write(content)

        print("Temp audio saved")

        # =========================
        # Fast ASR
        # =========================
        asr_result = transcribe_audio(
            temp_file_path
        )

        english_text = asr_result.get(
            "text",
            ""
        ).strip()

        # =========================
        # Empty Speech Check
        # =========================
        if not english_text:

            return {
                "status": "failed",
                "message": "No speech detected"
            }

        print(f"ASR Output: {english_text}")

        # =========================
        # Lightweight Refinement
        # =========================
        refined = refine_english(
            english_text
        )

        # =========================
        # Reverse Translation Disabled
        # Faster Processing
        # =========================
        reverse_result = {
            "malayalam": "",
            "status": "disabled"
        }

        # =========================
        # Save Record
        # =========================
        try:

            record = AudioRecord(
                filename=file.filename,
                language=asr_result.get(
                    "language",
                    "ml"
                ),
                transcript=english_text,
                translation=refined,
                malayalam_output=""
            )

            with Session(engine) as session:

                session.add(record)

                session.commit()

            print("Database save success")

        except Exception as db_error:

            print(
                f"Database save error: {db_error}"
            )

        # =========================
        # Final Response
        # =========================
        return {

            "status": "success",

            "asr_output": asr_result,

            "english_text": english_text,

            "refined_text": refined,

            "malayalam_text": ""

        }

    except Exception as e:

        print(f"Audio process error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # =========================
        # Cleanup Temp File
        # =========================
        if (
            temp_file_path
            and os.path.exists(temp_file_path)
        ):

            os.remove(temp_file_path)

            print("Temp file deleted")