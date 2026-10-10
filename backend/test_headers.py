import os, json, ssl, urllib.request
from dotenv import load_dotenv
load_dotenv('.env')

key = os.getenv("GEMINI_API_KEY", "").strip()
print("KEY:", key[:15])

ctx = ssl._create_unverified_context()
url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
body = json.dumps({"contents": [{"parts": [{"text": "Reply JSON: {\"status\": \"ok\"}"}]}]}).encode('utf-8')

headers_to_test = [
    {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    {"Content-Type": "application/json", "x-goog-api-key": key},
    {"Content-Type": "application/json", "api-key": key},
]

for i, h in enumerate(headers_to_test):
    req = urllib.request.Request(url, data=body, headers=h)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"HEADER TEST {i} SUCCESS:", txt.strip())
            break
    except Exception as e:
        print(f"HEADER TEST {i} FAILED:", e)
