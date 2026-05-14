from fastapi import APIRouter, UploadFile, File
import tempfile

from app.services.asr_service import transcribe_audio
from app.services.translation_service import translate_text_dummy
from app.services.translation_service import translate_eng_to_ml

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

    return {
        "asr_output": asr_result,
        "translation_output": translation_result,
        "reverse_translation": translate_eng_to_ml(translation_result["refined"])
    }