from fastapi import APIRouter, UploadFile, File
from sqlmodel import Session
import tempfile

from app.services.asr_service import transcribe_audio
from app.services.translation_service import translate_text_dummy, translate_eng_to_ml
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

    # Speech-to-Text (ASR)
    asr_result = transcribe_audio(temp_file_path)

    # Get text from segments if main text is empty
    transcript_text = asr_result["text"]
    if not transcript_text and asr_result.get("segments"):
        transcript_text = " ".join([seg["text"] for seg in asr_result["segments"]])

    # Translation
    translation_result = translate_text_dummy(transcript_text)
    reverse_result = translate_eng_to_ml(translation_result["refined"])

    # Save to database
    try:
        record = AudioRecord(
            filename=file.filename,
            language=asr_result.get("language", "unknown"),
            transcript=transcript_text,
            translation=translation_result.get("translation", ""),
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