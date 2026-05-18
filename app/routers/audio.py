from fastapi import APIRouter, UploadFile, File
from sqlmodel import Session
import tempfile

from app.services.asr_service import transcribe_audio
from app.services.translation_service import translate_eng_to_ml, refine_english
from app.models.audio_record import AudioRecord
from app.core.db import engine

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/process")
async def process_audio(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        temp_file_path = tmp.name
        content = await file.read()
        tmp.write(content)

    # Speech-to-Text + Translation (WhisperX task="translate")
    asr_result = transcribe_audio(temp_file_path)

    # WhisperX already translated to English
    english_text = asr_result.get("text", "")
    if not english_text and asr_result.get("segments"):
        english_text = " ".join([seg["text"] for seg in asr_result["segments"]])
    english_text = english_text.strip()

    # Use english_text for display
    transcript_text = english_text

    refined = refine_english(english_text)

    translation_result = {
        "transliteration": transcript_text,
        "translation": english_text,
        "refined": refined,
        "status": "success"
    }

    # Reverse: English back to Malayalam Unicode
    reverse_result = translate_eng_to_ml(refined)

    # Save to database
    try:
        record = AudioRecord(
            filename=file.filename,
            language=asr_result.get("language", "ml"),
            transcript=transcript_text,
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
