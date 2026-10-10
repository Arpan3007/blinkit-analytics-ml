import os, json, ssl, urllib.request
from dotenv import load_dotenv
load_dotenv('.env')

key = os.getenv("GEMINI_API_KEY", "").strip()

ctx = ssl._create_unverified_context()
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={key}"

prompt = """
Analyze this retail sales ML model result and respond with pure JSON:
Target: Sales_Amount, Algorithm: Random Forest, Accuracy: 94.2%

JSON format:
{
  "executive_summary": "High accuracy model for Sales Amount",
  "sales_insights": [{"title": "Strong Predictability", "insight": "Random forest captured 94.2% accuracy", "impact": "High", "action": "Deploy model"}],
  "product_insights": [],
  "client_insights": [],
  "model_quality": {"verdict": "Excellent", "confidence": "94.2%", "key_strength": "High precision", "key_weakness": "None", "improvement": "Add more features"},
  "predictions_summary": {"headline": "Reliable predictions", "details": "High confidence forecast"},
  "top_recommendations": [{"priority": 1, "title": "Scale inventory", "detail": "Use model to optimize stock"}]
}
"""

body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        txt = data["candidates"][0]["content"]["parts"][0]["text"]
        print("GEMINI FLASH LATEST SUCCESS!")
        print(txt[:300])
except Exception as e:
    print("GEMINI ERROR:", e)
