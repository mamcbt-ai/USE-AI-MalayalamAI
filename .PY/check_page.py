path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\malayalam-ai-frontend\app\page.js'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Check what fetch call looks like now
import re
m = re.search(r'sendAudioStream.*?finally.*?\}', c, re.DOTALL)
if m:
    print("FOUND FUNCTION:")
    print(m.group(0)[:600])
else:
    print("Function not found")
    idx = c.find('sendAudioStream')
    print(c[idx:idx+400])
