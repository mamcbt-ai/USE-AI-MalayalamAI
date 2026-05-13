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

    # Translation
    translation_result = translate_text_dummy(asr_result["text"])

    return {
    "asr_output": asr_output,
    "translation_output": translation_output,
    "reverse_translation": translate_eng_to_ml(translation_output["refined"])
}
