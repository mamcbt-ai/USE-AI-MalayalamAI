path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\malayalam-ai-frontend\app\page.js'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Fix 1: find the actual label text in JSX (without quotes)
import re

# Replace the Unicode label - find it in JSX context
for old, new in [
    ('Malayalam (Unicode)', '{selectedLang==="ml"?"Malayalam (Unicode)":selectedLang==="ta"?"Tamil (Unicode)":selectedLang==="te"?"Telugu (Unicode)":selectedLang==="kn"?"Kannada (Unicode)":"Hindi (Unicode)"}'),
    ('MALAYALAM (UNICODE)', '{selectedLang==="ml"?"MALAYALAM (UNICODE)":selectedLang==="ta"?"TAMIL (UNICODE)":selectedLang==="te"?"TELUGU (UNICODE)":selectedLang==="kn"?"KANNADA (UNICODE)":"HINDI (UNICODE)"}'),
    ("'MALAYALAM (UNICODE)'", 'selectedLang==="ml"?"MALAYALAM (UNICODE)":selectedLang==="ta"?"TAMIL (UNICODE)":selectedLang==="te"?"TELUGU (UNICODE)":selectedLang==="kn"?"KANNADA (UNICODE)":"HINDI (UNICODE)"'),
]:
    if old in c:
        c = c.replace(old, new)
        print(f"Replaced: {old[:30]}")

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Label fix done')
