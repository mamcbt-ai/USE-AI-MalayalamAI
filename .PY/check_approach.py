path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend\app\services\asr_service.py'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Whisper sometimes works better with transcription first, then LLM translation
old = '''def _groq_translate_to_english(wav_bytes: bytes) -> str:
    """Whisper large-v3 translations -> English"""
    try:
        result = groq_client.audio.translations.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            response_format="text",
        )'''

new = '''def _groq_translate_to_english(wav_bytes: bytes, lang: str = "ml") -> str:
    """First transcribe in native language, then use LLM to translate to English"""
    try:
        # Step 1: Transcribe in native language (more accurate than direct translation)
        result = groq_client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=WHISPER_MODEL,
            language=lang,
            response_format="text",
        )'''

c = c.replace(old, new)

# Pass lang to the function
c = c.replace(
    'english_text = _groq_translate_to_english(wav_bytes)\n        print(f"[ASR] English: {english_text}")\n\n        native_text = _llm_to_native(english_text, source_lang)',
    'native_text_raw = _groq_translate_to_english(wav_bytes, source_lang)\n        print(f"[ASR] Transcribed: {native_text_raw}")\n        # Use LLM to translate native transcription to English\n        english_text = _llm_translate_to_english(native_text_raw, source_lang) if native_text_raw else ""\n        print(f"[ASR] English: {english_text}")\n\n        native_text = native_text_raw  # Use direct transcription for native box'
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Done - but need to add _llm_translate_to_english function')
