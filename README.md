# InsightForge Analytics — DWM + ML Project

A professional full-stack data analytics and machine-learning workspace designed for the project:

**Sales and Customer Analysis of Blinkit using Python with Random Forest Prediction**

It is deliberately reusable: the Dashboard accepts a CSV and uses it as the active analytics dataset. ML Studio trains models from that active dataset, saves the selected best model, and can then accept a separate compatible CSV for prediction.

## Stack
- Frontend: React + TypeScript + Vite
- Backend: Python + FastAPI
- Data: Pandas + NumPy
- ML: scikit-learn
- Charts: Recharts
- Icons: Lucide React
- Optional database: not required for the current MVP; uploaded datasets and model artifacts are stored locally.

## Features
- Modern responsive analytics dashboard
- Import CSV and automatic schema detection
- Dataset health, dimensions and preview
- Sales / Customer / Product / Outlet analysis modules using dataset-aware generic analytics
- Preprocessing inspection
- ML Studio with regression/classification task selection
- Model comparison: Random Forest, Decision Tree, Linear/Logistic Regression
- Best-model selection based on held-out test metrics
- Random Forest feature importance when available
- Saved model artifact with joblib
- Separate prediction CSV workflow using the trained model
- Four themes: Sunny Yellow, Midnight, Royal Purple, Ruby
- Responsive desktop/tablet/mobile UI

## Important ML behavior
The Dashboard dataset and Prediction dataset are intentionally separate.

1. Upload the main Blinkit CSV on Dashboard.
2. Navigate to ML Studio.
3. Select the target column and task.
4. Train and compare models.
5. The best model is saved in `backend/models/`.
6. Upload a second CSV in the Prediction section.
7. The second CSV must contain the features expected by the trained model. Its columns must be semantically compatible with the training data; an arbitrary Zomato file cannot automatically be used to predict a Blinkit-trained target.

## Run locally
You need Python 3.10+ and Node.js 18+.

### Terminal 1 — backend
Windows:
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend: http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs

### Terminal 2 — frontend
```powershell
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

The frontend already defaults to `http://127.0.0.1:8000/api`. To change it, create `frontend/.env`:
```env
VITE_API_URL=http://127.0.0.1:8000/api
```

## Quick demo
Use `sample_data/blinkit_demo.csv` to verify the dashboard upload flow.

For a useful Random Forest demonstration, use a larger dataset with enough rows and a meaningful target. The bundled demo is intentionally tiny and is primarily for UI smoke testing.

## API endpoints
- `GET /api/health`
- `POST /api/datasets/upload`
- `GET /api/datasets/{dataset_id}`
- `POST /api/ml/train`
- `POST /api/ml/predict`

## Production notes
This is a college-project-ready local application, not a hardened public SaaS deployment. Before exposing it publicly, add authentication, file scanning, rate limits, persistent database storage, job queues, resource limits, secure model loading, HTTPS, and stricter schema validation.
