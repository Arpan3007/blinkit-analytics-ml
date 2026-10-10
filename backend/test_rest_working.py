import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv('.env')
key = os.getenv('GEMINI_API_KEY', '')

for m in ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3.6-flash"]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
    body = json.dumps({"contents": [{"parts": [{"text": "Say hello in JSON format: {\"reply\": \"hello\"}"}]}]}).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"SUCCESS {m}: -> {txt.strip()}")
            break
    except Exception as e:
        print(f"FAILED {m}: -> {e}")
