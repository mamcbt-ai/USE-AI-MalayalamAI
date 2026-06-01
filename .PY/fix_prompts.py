path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend\app\services\asr_service.py'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Upgrade to better model
c = c.replace('LLM_MODEL = "llama-3.1-8b-instant"', 'LLM_MODEL = "llama-3.3-70b-versatile"')

# Better prompts - more accurate, natural translation
old_prompts = '''NATIVE_PROMPTS = {
    "ml": "Translate this English text to Malayalam Unicode script. Output ONLY Malayalam characters (മലയാളം). No English, no explanation, no transliteration.",
    "ta": "Translate this English text to Tamil Unicode script. Output ONLY Tamil characters (தமிழ்). No English, no explanation.",
    "te": "Translate this English text to Telugu Unicode script. Output ONLY Telugu characters (తెలుగు). No English, no explanation.",
    "kn": "Translate this English text to Kannada Unicode script. Output ONLY Kannada characters (ಕನ್ನಡ). No English, no explanation.",
    "hi": "Translate this English text to Hindi Devanagari script. Output ONLY Hindi characters (हिंदी). No English, no explanation.",
}'''

new_prompts = '''NATIVE_PROMPTS = {
    "ml": """You are an expert Malayalam translator. Translate the given English text into natural, conversational Malayalam Unicode script.
Rules:
- Output ONLY Malayalam Unicode characters
- Use natural spoken Malayalam (not overly formal)
- Include common slang and colloquial expressions where appropriate
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "ഹലോ, എങ്ങനെ ഉണ്ട്?"
Translate now:""",
    "ta": """You are an expert Tamil translator. Translate the given English text into natural, conversational Tamil Unicode script.
Rules:
- Output ONLY Tamil Unicode characters
- Use natural spoken Tamil including colloquial expressions
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "வணக்கம், எப்படி இருக்கீங்க?"
Translate now:""",
    "te": """You are an expert Telugu translator. Translate the given English text into natural, conversational Telugu Unicode script.
Rules:
- Output ONLY Telugu Unicode characters
- Use natural spoken Telugu including colloquial expressions
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "హలో, ఎలా ఉన్నారు?"
Translate now:""",
    "kn": """You are an expert Kannada translator. Translate the given English text into natural, conversational Kannada Unicode script.
Rules:
- Output ONLY Kannada Unicode characters
- Use natural spoken Kannada including colloquial expressions
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "ಹಲೋ, ಹೇಗಿದ್ದೀರಾ?"
Translate now:""",
    "hi": """You are an expert Hindi translator. Translate the given English text into natural, conversational Hindi Devanagari script.
Rules:
- Output ONLY Hindi Devanagari characters
- Use natural spoken Hindi including colloquial expressions
- Do NOT output English, Roman transliteration, or any explanation
- Example: "Hello, how are you?" -> "नमस्ते, कैसे हो?"
Translate now:""",
}'''

c = c.replace(old_prompts, new_prompts)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Done')
