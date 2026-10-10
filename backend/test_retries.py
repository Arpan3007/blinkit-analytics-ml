import os, json, ssl, urllib.request, time
from dotenv import load_dotenv
load_dotenv('.env')

key = os.getenv("GEMINI_API_KEY", "").strip()
ctx = ssl._create_unverified_context()

candidate_models = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-pro-latest"]
prompt = "Reply with JSON: {\"status\": \"ok\", \"ai\": \"working\"}"
body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')

for m in candidate_models:
    print(f"Trying model {m}...")
    for attempt in range(3):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                txt = data["candidates"][0]["content"]["parts"][0]["text"]
                print(f"SUCCESS {m} (attempt {attempt+1}): {txt.strip()}")
                exit(0)
        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {m}: {e}")
            time.sleep(2)
