# InsightForge Backend

## Run
Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API: http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs

The API stores uploaded CSV files in `uploads/` and trained model artifacts in `models/`.
