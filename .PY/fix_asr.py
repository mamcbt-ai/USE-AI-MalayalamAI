path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\backend\app\services\asr_service.py'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Remove the prompt parameter that gets echoed back
old = '            prompt=f"This is {LANG_NAMES.get(lang, lang)} speech including colloquial and slang expressions.",'
new = ''
c = c.replace(old, new)

# Also filter out prompt echoes in results
old2 = '    return cleanup_text(text.strip())\n    except Exception as e:\n        print(f"[ASR] Transcribe error: {e}")\n        return ""'
new2 = '''    result = cleanup_text(text.strip())
        # Filter out if Groq echoed back the prompt
        prompt_echo = f"This is {LANG_NAMES.get(lang, lang)} speech"
        if result.lower().startswith(prompt_echo.lower()[:20]):
            return ""
        return result
    except Exception as e:
        print(f"[ASR] Transcribe error: {e}")
        return ""'''
c = c.replace(old2, new2)

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('ASR fix done')
