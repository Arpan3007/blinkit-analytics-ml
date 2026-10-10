import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv('.env')

def _call_gemini_api(prompt: str):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("NO API KEY FOUND")
        return None
    
    models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3.6-flash"]
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                txt = data["candidates"][0]["content"]["parts"][0]["text"]
                print(f"SUCCESS USING MODEL: {m}")
                return txt
        except Exception as e:
            print(f"MODEL {m} ERROR: {e}")
    return None

res = _call_gemini_api("Return a json object: {\"status\": \"ok\", \"message\": \"Gemini API connected\"}")
print("RESULT:", res)
