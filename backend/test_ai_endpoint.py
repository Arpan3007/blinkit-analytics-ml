import urllib.request, json

url = "http://127.0.0.1:8000/api/ml/train"
body = json.dumps({"dataset_id": "dataset1_retail_q1.csv", "target": "Outlet_ID", "task": "classification"}).encode('utf-8')
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("TRAIN SUCCESS!")
        print("POWERED BY:", res.get("powered_by"))
        ai = res.get("ai_insights", {})
        print("SUMMARY:", ai.get("executive_summary"))
except Exception as e:
    print("TRAIN API ERROR:", e)
