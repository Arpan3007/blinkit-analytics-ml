import os, json, ssl, urllib.request, time
from dotenv import load_dotenv
load_dotenv('.env')

key = os.getenv("GEMINI_API_KEY", "").strip()
ctx = ssl._create_unverified_context()

models = ["gemini-1.5-flash-latest", "gemini-flash-latest", "gemini-pro-latest", "gemini-1.5-pro-latest"]

prompt = "Hello, return JSON: {\"status\": \"ok\"}"
body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')

for m in models:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"SUCCESS {m}: {txt.strip()}")
            break
    except Exception as e:
        print(f"FAILED {m}: {e}")
