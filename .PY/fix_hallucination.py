path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend\app\services\asr_service.py'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Stronger hallucination detection for both English and native
old = '''def _is_hallucination(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    from collections import Counter
    counts = Counter(text.replace(" ", ""))
    if counts:
        ratio = counts.most_common(1)[0][1] / max(len(text.replace(" ", "")), 1)
        if ratio > 0.6:
            return True
    bad = ["thank you", "thanks for watching", "subscribe", "music", "[music]", "subtitles"]
    lower = text.lower()
    if any(p in lower for p in bad) and len(text) < 30:
        return True
    return False'''

new = '''def _is_hallucination(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    from collections import Counter
    counts = Counter(text.replace(" ", ""))
    if counts:
        ratio = counts.most_common(1)[0][1] / max(len(text.replace(" ", "")), 1)
        if ratio > 0.6:
            return True
    # Whisper hallucination phrases
    bad = [
        "thank you for watching", "thanks for watching", "subscribe",
        "music", "[music]", "subtitles", "captions",
        "thank you", "hello and welcome", "welcome to my channel",
        "hello, welcome", "translated by", "translation by",
        "english is a language", "language of the language",
    ]
    lower = text.lower()
    if any(lower.startswith(p) or (p in lower and len(text) < 60) for p in bad):
        return True
    # Detect word repetition (e.g. "kar do kar do kar do")
    words = text.split()
    if len(words) >= 4:
        unique = len(set(w.lower() for w in words))
        if unique / len(words) < 0.5:
            return True
    return False'''

c = c.replace(old, new)

# Apply hallucination check to English too
old2 = '''        english_text = _groq_translate(wav_bytes, source_lang)
        print(f"[ASR] English  : {english_text}")

        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}'''

new2 = '''        english_text = _groq_translate(wav_bytes, source_lang)
        if _is_hallucination(english_text):
            english_text = ""
        print(f"[ASR] English  : {english_text}")

        if english_text:
            yield {"type": "english_segment", "text": english_text, "accumulated": english_text}'''

c = c.replace(old2, new2)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Done')
