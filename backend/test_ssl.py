import os, json, ssl, urllib.request
from dotenv import load_dotenv
load_dotenv('.env')

key = os.getenv("GEMINI_API_KEY", "").strip()
print("KEY:", key[:10] if key else "NONE")

ctx = ssl._create_unverified_context()

# Test models directly
models = ["models/gemini-2.5-flash", "models/gemini-flash-latest", "models/gemini-1.5-flash", "models/gemini-2.0-flash"]

for m in models:
    url = f"https://generativelanguage.googleapis.com/v1beta/{m}:generateContent?key={key}"
    body = json.dumps({"contents": [{"parts": [{"text": "Reply JSON: {\"status\": \"ok\"}"}]}]}).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"SUCCESS FOR {m}:", txt.strip())
            break
    except Exception as e:
        print(f"ERROR FOR {m}:", e)
