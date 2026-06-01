path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend\app\services\asr_service.py'
with open(path, encoding='utf-8') as f:
    c = f.read()

old = "def cleanup_text(text):\n    if not text: return ''\n    return re.sub(r'\\s+', ' ', text.strip()).strip()"

new = """def cleanup_text(text):
    if not text: return ''
    text = re.sub(r'\\s+', ' ', text.strip()).strip()
    # Remove repeated sentences
    sentences = [s.strip() for s in re.split(r'[.!?。।]', text) if s.strip()]
    seen = []
    for s in sentences:
        if s not in seen:
            seen.append(s)
    return '. '.join(seen).strip()"""

c = c.replace(old, new)
with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Done')