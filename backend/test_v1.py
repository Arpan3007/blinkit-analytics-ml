import os, json, ssl, urllib.request, time
from dotenv import load_dotenv
load_dotenv('.env')

key = os.getenv("GEMINI_API_KEY", "").strip()
ctx = ssl._create_unverified_context()

tests = [
    ("v1", "gemini-1.5-flash"),
    ("v1beta", "gemini-1.5-flash"),
    ("v1", "gemini-1.5-pro"),
    ("v1beta", "gemini-1.5-pro"),
    ("v1beta", "gemini-flash-latest"),
    ("v1beta", "gemini-pro-latest"),
    ("v1beta", "gemini-2.5-flash"),
]

prompt = "Hello, return JSON: {\"status\": \"ok\"}"
body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')

for ver, m in tests:
    url = f"https://generativelanguage.googleapis.com/{ver}/models/{m}:generateContent?key={key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"SUCCESS ({ver}, {m}): {txt.strip()}")
            break
    except Exception as e:
        print(f"FAILED ({ver}, {m}): {e}")
        time.sleep(1)
