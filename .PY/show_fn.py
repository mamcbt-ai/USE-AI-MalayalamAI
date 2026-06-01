path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\malayalam-ai-frontend\app\page.js'
with open(path, encoding='utf-8') as f:
    lines = f.readlines()

in_fn = False
for i, line in enumerate(lines):
    if 'const sendAudioStream' in line:
        in_fn = True
    if in_fn:
        print(f"{i+1}: {line}", end='')
    if in_fn and i > 0 and 'finally' in line:
        break
