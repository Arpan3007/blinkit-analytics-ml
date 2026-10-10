import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv('.env')
key = os.getenv('GEMINI_API_KEY', '')
print('KEY EXISTS:', bool(key), 'LEN:', len(key))

models_to_test = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]

for m in models_to_test:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
    body = json.dumps({"contents": [{"parts": [{"text": "Say hello in JSON format: {\"reply\": \"hello\"}"}]}]}).encode('utf-8')
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            print(f"MODEL {m}: SUCCESS -> {txt[:60]}")
            break
    except Exception as e:
        print(f"MODEL {m}: FAILED -> {e}")
