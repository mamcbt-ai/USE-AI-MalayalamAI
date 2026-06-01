path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\malayalam-ai-frontend\app\page.js'
with open(path, encoding='utf-8') as f:
    lines = f.readlines()

new_lines = lines[:132]  # keep lines 1-132 (0-indexed: 0-131)
new_lines.append('      setStreamStatus(\'Processing...\');\n')
new_lines.append('      const data = await res.json();\n')
new_lines.append('      setLoading(false);\n')
new_lines.append('      const eng = data.english_text || \'\';\n')
new_lines.append('      const mal = data.malayalam_text || \'\';\n')
new_lines.append('      if (!eng && !mal) {\n')
new_lines.append('        setError(\'No speech detected. Please speak clearly and try again.\');\n')
new_lines.append('      } else {\n')
new_lines.append('        setEnglishLive(eng);\n')
new_lines.append('        setMalayalamLive(mal);\n')
new_lines.append('        setRefinedText(data.refined_text || eng);\n')
new_lines.append('      }\n')
new_lines.append('      setStreamStatus(\'\');\n')
new_lines.append('      setIsDone(true);\n')
new_lines += lines[167:]  # keep from line 168 onwards

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Done - replaced lines 133-167')
