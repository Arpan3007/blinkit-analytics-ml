import os, json, urllib.request
from dotenv import load_dotenv
load_dotenv('.env')
key = os.getenv('GEMINI_API_KEY', '')

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
req = urllib.request.Request(url)
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("AVAILABLE MODELS:")
        for m in data.get('models', []):
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                print(" -", m['name'])
except Exception as e:
    print("FAILED TO LIST MODELS:", e)
