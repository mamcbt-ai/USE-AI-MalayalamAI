import os

def refine_english(text: str) -> str:
    if not text or not text.strip():
        return ""
    clean = text.strip()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("refine_english: no OPENAI_API_KEY - returning raw transcript")
        return clean
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Fix grammar and punctuation of English text transcribed from Malayalam speech. Return only the corrected text."},
                {"role": "user", "content": clean}
            ],
            max_tokens=600, temperature=0.2,
        )
        refined = response.choices[0].message.content.strip()
        return refined if refined else clean
    except Exception as e:
        print(f"refine_english error: {e}")
        return clean
