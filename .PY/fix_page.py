import re

path = r'C:\Users\User\Desktop\USE AI_MalayalamAI_8 MAY 2026\malayalam-ai-frontend\app\page.js'
with open(path, encoding='utf-8') as f:
    c = f.read()

# Fix 1: change URL
c = c.replace('/audio/process-stream', '/audio/process')

# Fix 2: replace entire SSE reader block with simple JSON fetch
old_block = """      const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop();
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const raw = line.slice(6).trim();
            if (!raw) continue;
            let evt;
            try { evt = JSON.parse(raw); } catch { continue; }
            if (evt.type === 'status') { setStreamStatus(evt.message); setLoading(false); }
            else if (evt.type === 'english_segment') { setEnglishLive(evt.accumulated); setStreamStatus('Transcribing English...'); setLoading(false); }
            else if (evt.type === 'malayalam_segment') { setMalayalamLive(evt.accumulated); setStreamStatus('Transcribing Malayalam...'); }
            else if (evt.type === 'error') { setStreamStatus('Error: ' + (evt.error || 'Transcription failed. Check API key.')); setIsDone(true); }
            else if (evt.type === 'complete') {
              const eng = evt.english_text || '';
              const mal = evt.malayalam_text || '';
              if (!eng && !mal) {
                setError('No speech detected. Please speak clearly and try again.');
              } else {
                setEnglishLive(eng);
                setMalayalamLive(mal);
                setRefinedText(evt.refined_text || eng);
              }
              setStreamStatus('');
              setIsDone(true);
            }
            else if (evt.type === 'error') { setError('Error: ' + evt.message); }
          }
        }"""

new_block = """      setStreamStatus('Processing...');
        const data = await res.json();
        setLoading(false);
        const eng = data.english_text || '';
        const mal = data.malayalam_text || '';
        if (!eng && !mal) {
          setError('No speech detected. Please speak clearly and try again.');
        } else {
          setEnglishLive(eng);
          setMalayalamLive(mal);
          setRefinedText(data.refined_text || eng);
        }
        setStreamStatus('');
        setIsDone(true);"""

if old_block in c:
    c = c.replace(old_block, new_block)
    print('SSE reader replaced')
else:
    print('WARNING: old block not found - check manually')

# Fix 3: dynamic language label
c = c.replace("'MALAYALAM (UNICODE)'", "selectedLang==='ml'?'MALAYALAM (UNICODE)':selectedLang==='ta'?'TAMIL (UNICODE)':selectedLang==='te'?'TELUGU (UNICODE)':selectedLang==='kn'?'KANNADA (UNICODE)':'HINDI (UNICODE)'")
print('Language label fixed')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('Done')
