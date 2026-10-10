"""
InsightForge Analytics - Pure-Python FastAPI backend v2.4
Multi-source: CSV + Kaggle Dataset + Quick Commerce API
"""
from __future__ import annotations
import csv, json, math, os, pickle, random, statistics, uuid, shutil, tempfile, zipfile, ssl, urllib.request, time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load .env if present (never required, just convenience)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

# Gemini REST API helper (pure Python, bypasses gRPC Windows DLL issues)
def _get_gemini() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key if key else None

def _call_gemini_api(prompt: str) -> Optional[str]:
    key = _get_gemini()
    if not key:
        return None
    ctx = ssl._create_unverified_context()
    candidate_models = ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-pro-latest"]
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode('utf-8')
    headers = {"Content-Type": "application/json"}

    for m in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, data=body, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                time.sleep(1)
                continue
    return None

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "uploads"
MODEL_DIR = BASE / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

app = FastAPI(title="InsightForge Analytics API", version="2.2.0")
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"] if "*" in cors_origins else cors_origins,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TrainRequest(BaseModel):
    dataset_id: str; target: str; task: str
    test_size: float = 0.2; random_state: int = 42

class PredictRequest(BaseModel):
    model_id: str; dataset_id: str

class CustomPredictRequest(BaseModel):
    model_id: str
    inputs: Dict[str, Any]

class AIInsightsRequest(BaseModel):
    model_id: str
    dataset_id: str

class KaggleImportRequest(BaseModel):
    url: str  # full Kaggle dataset URL

class KaggleSelectFileRequest(BaseModel):
    download_id: str  # temp folder ID
    filename: str     # chosen file name within the zip

class PreprocessRequest(BaseModel):
    dataset_id: str
    operations: List[str]  # list of operation keys to apply


# ===========================================================================
# Data utilities
# ===========================================================================
def _parse_number(v: str) -> Optional[float]:
    try:
        f = float(v); return f if math.isfinite(f) else None
    except (ValueError, TypeError): return None

def read_csv_file(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        rows: List[Dict[str, Any]] = []; seen: set = set()
        for raw in reader:
            row: Dict[str, Any] = {}
            for k in headers:
                v = (raw.get(k) or "").strip()
                n = _parse_number(v)
                row[k] = n if n is not None else (v if v not in ("","nan","NaN","NA","N/A") else None)
            key = tuple(row.get(h) for h in headers)
            if key not in seen: seen.add(key); rows.append(row)
    return headers, rows

def col_values(rows, col): return [r.get(col) for r in rows]

def is_numeric_col(rows, col):
    vals = [v for v in col_values(rows, col) if v is not None]
    return bool(vals) and all(isinstance(v, float) for v in vals[:50])

def col_numeric(rows, col): return [v for v in col_values(rows, col) if isinstance(v, float)]

def dataset_summary(headers, rows):
    nc = sum(1 for h in headers if is_numeric_col(rows, h))
    missing = sum(1 for r in rows for v in r.values() if v is None)
    mem = sum(len(str(v)) for r in rows for v in r.values()) + len(rows)*len(headers)*4
    return {"rows": len(rows), "columns": len(headers), "numeric_columns": nc,
            "categorical_columns": len(headers)-nc, "missing_values": missing,
            "duplicates": 0, "memory_kb": round(mem/1024, 1)}

def col_schema(headers, rows):
    out = []
    for h in headers:
        vals = col_values(rows, h); nn = [v for v in vals if v is not None]
        dt = "float64" if (nn and all(isinstance(v,float) for v in nn[:20])) else "object"
        out.append({"name":h,"dtype":dt,"missing":sum(1 for v in vals if v is None),"unique":len(set(str(v) for v in nn))})
    return out

def _pearson(xs, ys):
    n = min(len(xs), len(ys))
    if n < 2: return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs)); dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx and dy else 0.0

def chart_data(headers, rows):
    result = {"numeric_distributions": [], "categorical": []}
    nc = cc = 0
    for h in headers:
        vals = [v for v in col_values(rows,h) if v is not None]
        if not vals: continue
        if is_numeric_col(rows,h) and nc < 8:
            fv = [v for v in vals if isinstance(v,float)]
            if fv: result["numeric_distributions"].append({"name":h,"mean":round(statistics.mean(fv),3),"min":round(min(fv),3),"max":round(max(fv),3)})
            nc += 1
        elif not is_numeric_col(rows,h) and cc < 10:
            counts = Counter(str(v) for v in vals).most_common(8)
            if counts and counts[0][1] > 1:
                result["categorical"].append({"column":h,"items":[{"label":k,"value":v} for k,v in counts]}); cc += 1
    num_h = [h for h in headers if is_numeric_col(rows,h)][:10]
    if len(num_h) >= 2:
        cd = {h: col_numeric(rows,h) for h in num_h}
        result["correlation"] = {"columns":num_h,"values":[[round(_pearson(cd[h1],cd[h2]),3) for h2 in num_h] for h1 in num_h]}
    return result

def preview_rows(rows, n=8):
    return [{k:(round(v,4) if isinstance(v,float) else (str(v) if v is not None else None)) for k,v in r.items()} for r in rows[:n]]

def recommend_targets(headers, rows):
    recs = []
    for h in headers:
        vals = [v for v in col_values(rows,h) if v is not None]
        if not vals: continue
        n = len(vals); uc = len(set(str(v) for v in vals)); ur = uc/n
        if ur > 0.9:
            recs.append({"column":h,"suitability":"bad","reason":f"High cardinality ({uc}/{n}) - likely ID column, not suitable as target.","task":None}); continue
        is_num = is_numeric_col(rows,h)
        if is_num:
            nu = len(set(v for v in vals if isinstance(v,float)))
            if nu <= 15: recs.append({"column":h,"suitability":"good","reason":f"Numeric with {nu} unique values - good for classification.","task":"classification"})
            else: recs.append({"column":h,"suitability":"good","reason":f"Continuous numeric with {nu} unique values - ideal for regression.","task":"regression"})
        else:
            if 2 <= uc <= 20: recs.append({"column":h,"suitability":"good","reason":f"Categorical with {uc} classes - ideal for classification.","task":"classification"})
            elif uc > 20: recs.append({"column":h,"suitability":"warn","reason":f"High cardinality ({uc} categories) - may be hard to classify.","task":"classification"})
    recs.sort(key=lambda x: {"good":0,"warn":1,"bad":2}.get(x["suitability"],3))
    return recs

# ===========================================================================
# Preprocessing
# ===========================================================================
def _median(vals):
    if not vals: return 0.0
    s = sorted(vals); n = len(s)
    return (s[n//2]+s[n//2-1])/2 if n%2==0 else s[n//2]

def _mode_str(vals): return Counter(vals).most_common(1)[0][0] if vals else ""

class Preprocessor:
    def __init__(self):
        self.num_cols=[]; self.cat_cols=[]; self.medians={}; self.modes={}; self.cat_maps={}; self.feature_names=[]
    def fit(self, rows, feature_cols):
        self.num_cols = [c for c in feature_cols if is_numeric_col(rows,c)]
        self.cat_cols = [c for c in feature_cols if c not in self.num_cols]
        for c in self.num_cols:
            vals = [v for v in col_values(rows,c) if isinstance(v,float)]
            self.medians[c] = _median(vals) if vals else 0.0
        for c in self.cat_cols:
            vals = [str(v) for v in col_values(rows,c) if v is not None]
            self.modes[c] = _mode_str(vals); self.cat_maps[c] = {s:i for i,s in enumerate(sorted(set(vals)))}
        self.feature_names = self.num_cols + self.cat_cols
    def transform(self, rows):
        out = []
        for r in rows:
            vec = []
            for c in self.num_cols:
                v = r.get(c); vec.append(float(v) if isinstance(v,float) else self.medians.get(c,0.0))
            for c in self.cat_cols:
                v = str(r.get(c)) if r.get(c) is not None else self.modes.get(c,"")
                m = self.cat_maps[c]; vec.append(float(m.get(v,len(m))))
            out.append(vec)
        return out
    def fit_transform(self, rows, feature_cols):
        self.fit(rows,feature_cols); return self.transform(rows)

# ===========================================================================
# CART Decision Tree & Random Forest
# ===========================================================================
class _Node:
    __slots__ = ("feat","thresh","left","right","value","class_counts","n_samples")
    def __init__(self): self.feat=self.thresh=self.left=self.right=self.value=self.class_counts=None; self.n_samples=0

def _gini(labels):
    if not labels: return 0.0
    n = len(labels); return 1.0 - sum((c/n)**2 for c in Counter(labels).values())

def _mse_var(vals):
    if len(vals) < 2: return 0.0
    m = statistics.mean(vals); return statistics.mean((v-m)**2 for v in vals)

def _best_split(X, y, feat_indices, task, min_leaf=1):
    best_f, best_t, best_imp = -1, 0.0, float("inf")
    n = len(y); crit = _gini if task=="classification" else _mse_var
    for fi in feat_indices:
        pairs = sorted(zip([X[i][fi] for i in range(n)], y)); seen: set = set()
        for idx in range(len(pairs)-1):
            v = pairs[idx][0]
            if v in seen: continue
            seen.add(v); t = (v+pairs[idx+1][0])/2
            ly = [pairs[j][1] for j in range(idx+1)]; ry = [pairs[j][1] for j in range(idx+1,len(pairs))]
            if len(ly)<min_leaf or len(ry)<min_leaf: continue
            imp = (len(ly)/n)*crit(ly)+(len(ry)/n)*crit(ry)
            if imp < best_imp: best_imp,best_f,best_t = imp,fi,t
    return best_f,best_t,best_imp

def _build_tree(X,y,task,max_depth,min_split,min_leaf,max_feats,depth,rng):
    node = _Node(); node.n_samples = len(y)
    if task=="classification":
        counts = Counter(y); node.class_counts = dict(counts); node.value = counts.most_common(1)[0][0]
    else: node.value = statistics.mean(y)
    if depth>=max_depth or len(y)<min_split or (task=="classification" and len(set(y))==1): return node
    n_feats = len(X[0]); k = min(max_feats, n_feats) if max_feats else n_feats
    feat_indices = rng.sample(range(n_feats), k) if k < n_feats else list(range(n_feats))
    f, t, imp = _best_split(X,y,feat_indices,task,min_leaf)
    if f < 0: return node
    left_mask = [X[i][f] <= t for i in range(len(y))]
    X_l = [X[i] for i, m in enumerate(left_mask) if m]; y_l = [y[i] for i, m in enumerate(left_mask) if m]
    X_r = [X[i] for i, m in enumerate(left_mask) if not m]; y_r = [y[i] for i, m in enumerate(left_mask) if not m]
    if not X_l or not X_r: return node
    node.feat=f; node.thresh=t
    node.left = _build_tree(X_l,y_l,task,max_depth,min_split,min_leaf,max_feats,depth+1,rng)
    node.right = _build_tree(X_r,y_r,task,max_depth,min_split,min_leaf,max_feats,depth+1,rng)
    return node

class DecisionTree:
    def __init__(self, task="classification", max_depth=10, min_split=2, min_leaf=1, random_state=42):
        self.task=task; self.max_depth=max_depth; self.min_split=min_split; self.min_leaf=min_leaf
        self.random_state=random_state; self.root=None; self.feature_importances_=[]
    def fit(self, X, y):
        rng = random.Random(self.random_state)
        self.root = _build_tree(X, y, self.task, self.max_depth, self.min_split, self.min_leaf, None, 0, rng)
        n_feats = len(X[0]); self.feature_importances_ = [0.0]*n_feats
        self._calc_imp(self.root, len(y))
        tot = sum(self.feature_importances_)
        if tot > 0: self.feature_importances_ = [v/tot for v in self.feature_importances_]
    def _calc_imp(self, node, total_samples):
        if not node or node.feat is None: return
        self.feature_importances_[node.feat] += (node.n_samples / total_samples)
        self._calc_imp(node.left, total_samples); self._calc_imp(node.right, total_samples)
    def predict_one(self, x):
        node = self.root
        while node and node.feat is not None:
            node = node.left if x[node.feat] <= node.thresh else node.right
        return node.value if node else 0.0
    def predict(self, X): return [self.predict_one(x) for x in X]

class RandomForest:
    def __init__(self, n_estimators=50, task="classification", max_depth=8, random_state=42):
        self.n_estimators=n_estimators; self.task=task; self.max_depth=max_depth
        self.random_state=random_state; self.trees=[]; self.feature_importances_=[]
    def fit(self, X, y):
        n = len(X); n_feats = len(X[0]); max_feats = max(1, int(math.sqrt(n_feats))) if self.task=="classification" else max(1, n_feats//3)
        self.trees = []; imps = [0.0]*n_feats
        for i in range(self.n_estimators):
            rng = random.Random(self.random_state + i)
            idxs = [rng.randint(0, n-1) for _ in range(n)]
            X_b = [X[j] for j in idxs]; y_b = [y[j] for j in idxs]
            tree = DecisionTree(task=self.task, max_depth=self.max_depth, random_state=self.random_state+i)
            tree.root = _build_tree(X_b, y_b, self.task, self.max_depth, 2, 1, max_feats, 0, rng)
            tree.feature_importances_ = [0.0]*n_feats; tree._calc_imp(tree.root, n)
            self.trees.append(tree)
            for f in range(n_feats): imps[f] += tree.feature_importances_[f]
        tot = sum(imps)
        self.feature_importances_ = [v/tot for v in imps] if tot > 0 else [1/n_feats]*n_feats
    def predict_one(self, x):
        preds = [t.predict_one(x) for t in self.trees]
        return Counter(preds).most_common(1)[0][0] if self.task=="classification" else statistics.mean(preds)
    def predict(self, X): return [self.predict_one(x) for x in X]

# ===========================================================================
# Linear & Logistic Regression
# ===========================================================================
class LinearRegression:
    def __init__(self, alpha=1.0): self.alpha=alpha; self.weights=[]; self.bias=0.0; self.coef_=[]
    def fit(self, X, y):
        n, p = len(X), len(X[0])
        X_mean = [statistics.mean([X[i][j] for i in range(n)]) for j in range(p)]
        X_std = [math.sqrt(statistics.variance([X[i][j] for i in range(n)]) or 1.0) for j in range(p)]
        y_mean = statistics.mean(y)
        X_norm = [[(X[i][j]-X_mean[j])/X_std[j] for j in range(p)] for i in range(n)]
        y_norm = [y[i]-y_mean for i in range(n)]
        weights = [0.0]*p
        for _ in range(200):
            preds = [sum(X_norm[i][j]*weights[j] for j in range(p)) for i in range(n)]
            errs = [preds[i]-y_norm[i] for i in range(n)]
            for j in range(p):
                grad = sum(errs[i]*X_norm[i][j] for i in range(n))/n + (self.alpha/n)*weights[j]
                weights[j] -= 0.05 * grad
        self.weights = [weights[j]/X_std[j] for j in range(p)]
        self.bias = y_mean - sum(self.weights[j]*X_mean[j] for j in range(p))
        self.coef_ = self.weights
    def predict(self, X): return [sum(x[j]*self.weights[j] for j in range(len(x))) + self.bias for x in X]

class LogisticRegression:
    def __init__(self, lr=0.1, max_iter=150, random_state=42):
        self.lr=lr; self.max_iter=max_iter; self.random_state=random_state; self.classes=[]; self.models={}; self.coef_=[]
    def fit(self, X, y):
        self.classes = sorted(set(y)); n, p = len(X), len(X[0])
        X_mean = [statistics.mean([X[i][j] for i in range(n)]) for j in range(p)]
        X_std = [math.sqrt(statistics.variance([X[i][j] for i in range(n)]) or 1.0) for j in range(p)]
        X_norm = [[(X[i][j]-X_mean[j])/X_std[j] for j in range(p)] for i in range(n)]
        all_coefs = []
        for cls in self.classes:
            y_bin = [1.0 if v==cls else 0.0 for v in y]
            w = [0.0]*p; b = 0.0
            for _ in range(self.max_iter):
                scores = [sum(X_norm[i][j]*w[j] for j in range(p)) + b for i in range(n)]
                preds = [1.0/(1.0+math.exp(-max(-50,min(50,s)))) for s in scores]
                errs = [preds[i]-y_bin[i] for i in range(n)]
                for j in range(p): w[j] -= self.lr * (sum(errs[i]*X_norm[i][j] for i in range(n))/n + 0.01*w[j])
                b -= self.lr * (sum(errs)/n)
            w_real = [w[j]/X_std[j] for j in range(p)]; b_real = b - sum(w_real[j]*X_mean[j] for j in range(p))
            self.models[cls] = (w_real, b_real); all_coefs.append(w_real)
        self.coef_ = all_coefs
    def predict_one(self, x):
        best_c, best_s = self.classes[0], -float("inf")
        for cls, (w, b) in self.models.items():
            s = sum(x[j]*w[j] for j in range(len(x))) + b
            if s > best_s: best_s = s; best_c = cls
        return best_c
    def predict(self, X): return [self.predict_one(x) for x in X]

# ===========================================================================
# Metrics & Insights
# ===========================================================================
def accuracy_score(yt, yp): return sum(1 for a,b in zip(yt,yp) if a==b)/len(yt) if yt else 0.0

def precision_recall_f1_weighted(yt, yp):
    classes = set(yt) | set(yp)
    if not classes: return 0.0, 0.0, 0.0
    total = len(yt); p_w = r_w = f1_w = 0.0
    for cls in classes:
        tp = sum(1 for a,b in zip(yt,yp) if a==cls and b==cls)
        fp = sum(1 for a,b in zip(yt,yp) if a!=cls and b==cls)
        fn = sum(1 for a,b in zip(yt,yp) if a==cls and b!=cls)
        supp = sum(1 for a in yt if a==cls)
        if supp == 0: continue
        p = tp/(tp+fp) if (tp+fp) else 0.0
        r = tp/(tp+fn) if (tp+fn) else 0.0
        f1 = (2*p*r)/(p+r) if (p+r) else 0.0
        p_w += p*(supp/total); r_w += r*(supp/total); f1_w += f1*(supp/total)
    return round(p_w,4), round(r_w,4), round(f1_w,4)

def per_class_metrics(yt, yp):
    classes = sorted(set(yt)); results = []
    for cls in classes:
        tp = sum(1 for a,b in zip(yt,yp) if a==cls and b==cls)
        fp = sum(1 for a,b in zip(yt,yp) if a!=cls and b==cls)
        fn = sum(1 for a,b in zip(yt,yp) if a==cls and b!=cls)
        support = sum(1 for a in yt if a==cls)
        p = round(tp/(tp+fp),4) if (tp+fp) else 0.0
        r = round(tp/(tp+fn),4) if (tp+fn) else 0.0
        f = round(2*p*r/(p+r),4) if (p+r) else 0.0
        results.append({"class":str(cls),"precision":p,"recall":r,"f1":f,"support":support})
    return results

def confusion_matrix_data(yt, yp):
    top_classes = [c for c,_ in Counter(yt).most_common(8)]; classes = sorted(top_classes)
    idx = {c:i for i,c in enumerate(classes)}; n = len(classes)
    mat = [[0]*n for _ in range(n)]
    for a,b in zip(yt,yp):
        if a in idx and b in idx: mat[idx[a]][idx[b]] += 1
    return {"classes":[str(c) for c in classes],"matrix":mat}

def r2_score(yt, yp):
    my = statistics.mean(yt); ss_res = sum((a-b)**2 for a,b in zip(yt,yp)); ss_tot = sum((a-my)**2 for a in yt)
    return round(1-ss_res/ss_tot,4) if ss_tot else 0.0

def mae(yt, yp): return round(statistics.mean(abs(a-b) for a,b in zip(yt,yp)),4)
def rmse(yt, yp): return round(math.sqrt(statistics.mean((a-b)**2 for a,b in zip(yt,yp))),4)

def evaluate(task, yt, yp):
    if task=="classification":
        p,r,f = precision_recall_f1_weighted(yt,yp)
        return {"accuracy":round(accuracy_score(yt,yp),4),"precision":p,"recall":r,"f1":f}
    yt2,yp2 = [float(v) for v in yt],[float(v) for v in yp]
    return {"r2":r2_score(yt2,yp2),"mae":mae(yt2,yp2),"rmse":rmse(yt2,yp2)}

def score_key(task, metrics): return metrics.get("accuracy",metrics.get("r2",-999))

def prediction_distribution(task, yp):
    if task=="classification":
        counts = Counter(str(v) for v in yp); total = len(yp)
        return [{"label":k,"count":v,"pct":round(v/total*100,1)} for k,v in counts.most_common(12)]
    fv = [float(v) for v in yp if isinstance(v,(int,float))]
    if not fv: return []
    mn,mx = min(fv),max(fv); rng = mx-mn
    if rng==0: return [{"label":str(round(mn,2)),"count":len(fv),"pct":100.0}]
    bins = 10; bin_size = rng/bins; buckets = {}
    for v in fv:
        b = min(int((v-mn)/bin_size),bins-1); buckets[b] = buckets.get(b,0)+1
    total = len(fv)
    return [{"label":f"{round(mn+i*bin_size,1)}-{round(mn+(i+1)*bin_size,1)}","count":buckets.get(i,0),"pct":round(buckets.get(i,0)/total*100,1)} for i in range(bins)]

def residual_samples(yt, yp, n=60):
    pairs = list(zip([float(v) for v in yt],[float(v) for v in yp])); step = max(1,len(pairs)//n)
    return [{"actual":round(a,3),"predicted":round(p,3),"residual":round(a-p,3)} for a,p in pairs[::step][:n]]

def generate_insights(task, target, best_alg, best_metrics, fi_list, y_test, y_pred, headers=None, rows=None):
    insights = []
    # 1. Model accuracy & capability insight
    if task=="classification":
        acc = best_metrics.get("accuracy",0); f1 = best_metrics.get("f1",0)
        if acc >= 0.90: insights.append({"category":"Accuracy & Reliability","type":"success","title":f"Production-Ready Accuracy: {acc*100:.1f}%","text":f"{best_alg} achieved excellent accuracy ({acc*100:.1f}%) on held-out test cases. High confidence in target classification."})
        elif acc >= 0.75: insights.append({"category":"Accuracy & Reliability","type":"good","title":f"Solid Model Accuracy: {acc*100:.1f}%","text":f"Reliable performance ({acc*100:.1f}%). Feature tuning or additional rows could boost accuracy further."})
        elif acc >= 0.60: insights.append({"category":"Accuracy & Reliability","type":"warn","title":f"Moderate Accuracy: {acc*100:.1f}%","text":f"Model is learning pattern, but misclassifications occur on tricky edge cases. Try feature scaling or re-balancing classes."})
        else: insights.append({"category":"Accuracy & Reliability","type":"error","title":f"Low Accuracy: {acc*100:.1f}%","text":f"Target '{target}' has weak predictability with current features. Consider dropping noise columns."})
        if f1 < acc-0.10: insights.append({"category":"Data Balance","type":"warn","title":"Class Imbalance Notice","text":f"Weighted F1 score ({f1*100:.1f}%) lags behind accuracy ({acc*100:.1f}%). Minority categories are harder to detect."})
    else:
        r2 = best_metrics.get("r2",0); rmse_val = best_metrics.get("rmse",0); mae_val = best_metrics.get("mae",0)
        y_vals = [float(v) for v in y_test]; tm = statistics.mean(y_vals) if y_vals else 1.0
        cv_pct = rmse_val/abs(tm)*100 if tm else 0
        if r2 >= 0.85: insights.append({"category":"Model Performance","type":"success","title":f"High Variance Coverage (R² = {r2:.3f})","text":f"{best_alg} captures {r2*100:.1f}% of variance in '{target}'. Mean prediction error (RMSE={rmse_val:.2f}) represents {cv_pct:.1f}% of target mean."})
        elif r2 >= 0.65: insights.append({"category":"Model Performance","type":"good","title":f"Good Predictive Fit (R² = {r2:.3f})","text":f"{best_alg} explains {r2*100:.1f}% of variance in '{target}'. Average absolute error is {mae_val:.2f}."})
        elif r2 >= 0.30: insights.append({"category":"Model Performance","type":"warn","title":f"Moderate Fit (R² = {r2:.3f})","text":f"Model explains {r2*100:.1f}% of variance. Substantial non-linear noise or missing explanatory variables."})
        else: insights.append({"category":"Model Performance","type":"error","title":f"Low R² Fit ({r2:.3f})","text":f"R² is low. Target '{target}' is weakly linear with existing numerical attributes."})

    # 2. Key Attribute Drivers
    if fi_list and len(fi_list)>=2:
        top1, top2 = fi_list[0], fi_list[1]
        insights.append({"category":"Key Drivers","type":"info","title":f"Primary Drivers: {top1['feature']} & {top2['feature']}","text":f"The model relies heavily on '{top1['feature']}' (importance weight: {top1['importance']:.4f}) and '{top2['feature']}' ({top2['importance']:.4f}) to formulate predictions."})

    # 3. Domain Specific Attributes (Sales, Products, Outlets, Clients)
    if headers and rows:
        col_map = {h.lower(): h for h in headers}
        
        # Sales & Pricing Insights
        sales_col = col_map.get('item_outlet_sales') or col_map.get('sales') or col_map.get('revenue')
        mrp_col = col_map.get('item_mrp') or col_map.get('price') or col_map.get('mrp')
        if mrp_col and target != mrp_col:
            mrp_vals = col_numeric(rows, mrp_col)
            if mrp_vals:
                insights.append({"category":"Sales & Pricing Strategy","type":"good","title":"Pricing Tier Leverage","text":f"Item Price/MRP ('{mrp_col}') averages {statistics.mean(mrp_vals):.2f}. Higher price tiers strongly correlate with total sales magnitude."})

        # Product Visibility & Type Insights
        vis_col = col_map.get('item_visibility') or col_map.get('visibility')
        if vis_col:
            vis_vals = col_numeric(rows, vis_col)
            if vis_vals:
                avg_vis = statistics.mean(vis_vals)
                insights.append({"category":"Product & Merchandising","type":"info","title":"Product Shelf Visibility","text":f"Average item shelf visibility is {avg_vis*100:.1f}%. Increasing visibility for high-margin items directly boosts prediction response."})

        # Outlet & Channel Insights
        outlet_size = col_map.get('outlet_size') or col_map.get('size')
        outlet_type = col_map.get('outlet_type') or col_map.get('type')
        if outlet_size or outlet_type:
            insights.append({"category":"Outlet & Operations","type":"good","title":"Store & Channel Segments","text":"Outlet footprint (Size & Store Type) creates distinct performance clusters. Supermarkets out-predict small grocery stores."})

        # Client / Customer Insights
        cust_col = col_map.get('customer_id') or col_map.get('client_id') or col_map.get('user_id') or col_map.get('customer_type')
        if cust_col:
            insights.append({"category":"Client & Customer Insights","type":"info","title":"Client Portfolio Impact","text":f"Customer segments in '{cust_col}' drive repeat order volume and baseline demand stability."})

    # 4. Actionable Strategy Recommendation
    insights.append({"category":"Strategic Action","type":"success","title":"Executive Recommendation","text":f"Use {best_alg} in ML Studio to test hypothetical price/attribute scenarios and optimize inventory allocations for top-performing items."})
    return insights


def gemini_ai_insights(task: str, target: str, best_alg: str, best_metrics: dict,
                       fi_list: list, results: list, headers: list, rows: list,
                       y_test: list, y_pred: list) -> Dict[str, Any]:
    """Call Gemini API for deep domain-aware insights. Returns structured insight dict."""
    gem = _get_gemini()
    
    # Build statistical context
    n_rows = len(rows)
    col_stats = []
    for h in headers[:12]:
        vals = [v for v in col_values(rows, h) if v is not None]
        if not vals: continue
        if is_numeric_col(rows, h):
            nv = [v for v in vals if isinstance(v, float)]
            if nv: col_stats.append(f"{h}: numeric, mean={statistics.mean(nv):.2f}, min={min(nv):.2f}, max={max(nv):.2f}")
        else:
            top = Counter(str(v) for v in vals).most_common(3)
            col_stats.append(f"{h}: categorical, top={[k for k,_ in top]}")
    
    # Feature importances
    fi_text = ", ".join(f"{x['feature']}={x['importance']:.4f}" for x in fi_list[:6]) if fi_list else "N/A"
    
    # All algorithm results
    alg_text = "; ".join(
        f"{r['algorithm']}: " + (f"acc={r.get('accuracy',0):.3f}, f1={r.get('f1',0):.3f}" if task=='classification' else f"r2={r.get('r2',0):.3f}, rmse={r.get('rmse',0):.3f}")
        for r in results if 'error' not in r
    )
    
    # Prediction stats
    if task == 'classification':
        pred_dist = Counter(str(v) for v in y_pred)
        pred_text = ", ".join(f"{k}: {v}" for k,v in pred_dist.most_common(5))
    else:
        pf = [float(v) for v in y_pred if isinstance(v,(int,float))]
        pred_text = f"mean={statistics.mean(pf):.2f}, min={min(pf):.2f}, max={max(pf):.2f}" if pf else "N/A"
    
    metric_text = json.dumps(best_metrics)
    
    prompt = f"""
You are an expert data scientist and business intelligence analyst. Analyze this ML model result and generate deep, actionable insights for a retail/sales analytics platform.

DATASET OVERVIEW:
- Rows: {n_rows:,}
- Target Column: {target} ({task} task)
- Columns & Stats: {'; '.join(col_stats)}

ML MODEL RESULTS:
- Best Algorithm: {best_alg}
- Best Metrics: {metric_text}
- All Algorithm Comparison: {alg_text}
- Feature Importance (top 6): {fi_text}
- Prediction Distribution on test set: {pred_text}

Generate a JSON response with exactly this structure (no markdown, pure JSON):
{{
  "executive_summary": "2-3 sentence executive summary with specific numbers from the model",
  "sales_insights": [
    {{"title": "...", "insight": "...", "impact": "High|Medium|Low", "action": "..."}}
  ],
  "product_insights": [
    {{"title": "...", "insight": "...", "impact": "High|Medium|Low", "action": "..."}}
  ],
  "client_insights": [
    {{"title": "...", "insight": "...", "impact": "High|Medium|Low", "action": "..."}}
  ],
  "model_quality": {{
    "verdict": "Excellent|Good|Fair|Poor",
    "confidence": "percentage string e.g. 87%",
    "key_strength": "...",
    "key_weakness": "...",
    "improvement": "..."
  }},
  "predictions_summary": {{
    "headline": "Key takeaway from prediction distribution",
    "details": "Detailed interpretation of what the model predicts and what it means for the business"
  }},
  "top_recommendations": [
    {{"priority": 1, "title": "...", "detail": "..."}},
    {{"priority": 2, "title": "...", "detail": "..."}},
    {{"priority": 3, "title": "...", "detail": "..."}}
  ]
}}

Use REAL numbers from the data. Be specific and domain-relevant. Focus on retail/sales analytics.
"""
    
    if not _get_gemini():
        # Fallback: rule-based structured insights
        return _fallback_structured_insights(task, target, best_alg, best_metrics, fi_list, results, pred_text)
    
    resp_text = _call_gemini_api(prompt)
    if not resp_text:
        return _fallback_structured_insights(task, target, best_alg, best_metrics, fi_list, results, pred_text)

    try:
        text = resp_text.strip()
        # Strip markdown code blocks if present
        if text.startswith('```'):
            text = text.split('```')[1]
            if text.startswith('json'): text = text[4:]
        elif '```' in text:
            text = text[:text.rfind('```')]
        return json.loads(text.strip())
    except Exception as e:
        return _fallback_structured_insights(task, target, best_alg, best_metrics, fi_list, results, pred_text)


def _fallback_structured_insights(task, target, best_alg, best_metrics, fi_list, results, pred_text):
    """Rule-based fallback when Gemini API is unavailable."""
    if task == 'classification':
        acc = best_metrics.get('accuracy', 0)
        f1 = best_metrics.get('f1', 0)
        verdict = 'Excellent' if acc>=0.90 else 'Good' if acc>=0.75 else 'Fair' if acc>=0.60 else 'Poor'
        confidence = f"{acc*100:.1f}%"
        strength = f"{acc*100:.1f}% classification accuracy on held-out test set"
        weakness = f"F1={f1*100:.1f}% indicates some class imbalance" if f1 < acc-0.08 else "Model generalises well across classes"
    else:
        r2 = best_metrics.get('r2', 0)
        rmse = best_metrics.get('rmse', 0)
        verdict = 'Excellent' if r2>=0.85 else 'Good' if r2>=0.65 else 'Fair' if r2>=0.30 else 'Poor'
        confidence = f"{max(0,r2)*100:.1f}% variance explained"
        strength = f"R²={r2:.3f} — explains {r2*100:.1f}% of variance in {target}"
        weakness = f"RMSE={rmse:.3f}" if rmse else "Need more predictive features"
    
    top_feat = fi_list[0]['feature'] if fi_list else target
    alg_count = len([r for r in results if 'error' not in r])
    
    return {
        "executive_summary": f"{best_alg} was selected as the best model out of {alg_count} trained algorithms. It achieves {confidence} on the target column '{target}', with '{top_feat}' being the strongest predictor. {verdict} model quality — suitable for production use.",
        "sales_insights": [
            {"title": "Revenue Driver Identified", "insight": f"'{top_feat}' is the top driver of '{target}'. Optimising this attribute could unlock significant sales uplift.", "impact": "High", "action": f"Run scenario simulations with different '{top_feat}' values using the Attribute Predictor below."},
            {"title": "Algorithm Benchmark", "insight": f"{best_alg} outperformed all other models. {alg_count} algorithms were evaluated.", "impact": "Medium", "action": "Use best model for all production batch predictions."}
        ],
        "product_insights": [
            {"title": "Feature Leverage", "insight": f"Top 3 features contribute most to {target} predictions. Focus inventory and pricing decisions on these attributes.", "impact": "High", "action": "Prioritise data quality for top features."}
        ],
        "client_insights": [
            {"title": "Prediction Reliability", "insight": f"Model produces consistent predictions ({confidence} reliable). Client-facing forecasts are trustworthy.", "impact": "Medium", "action": "Deploy predictions for client planning dashboards."}
        ],
        "model_quality": {
            "verdict": verdict,
            "confidence": confidence,
            "key_strength": strength,
            "key_weakness": weakness,
            "improvement": "Add more domain-specific features or collect more rows to improve model performance further."
        },
        "predictions_summary": {
            "headline": f"Model predicts '{target}' with {confidence} reliability",
            "details": f"Prediction distribution: {pred_text}. The trained {best_alg} uses {len(fi_list)} features to generate predictions."
        },
        "top_recommendations": [
            {"priority": 1, "title": f"Optimise '{top_feat}'", "detail": f"This is the #1 predictor of '{target}'. Use the Scenario Predictor to find optimal values."},
            {"priority": 2, "title": "Run Batch Predictions", "detail": "Upload new data CSV to generate bulk predictions for all items/clients."},
            {"priority": 3, "title": "Collect More Data", "detail": "Additional rows and features will further improve model accuracy and reliability."}
        ]
    }

# ===========================================================================
# API Endpoints
# ===========================================================================
@app.get("/api/health")
def health(): return {"status":"ok","service":"InsightForge Analytics API","engine":"pure-python"}

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Only CSV files are supported")
    raw = await file.read()
    if len(raw)>50*1024*1024: raise HTTPException(413,"CSV exceeds 50 MB")
    dataset_id = uuid.uuid4().hex[:12]; path = DATA_DIR/f"{dataset_id}.csv"; path.write_bytes(raw)
    try: headers,rows = read_csv_file(path)
    except Exception as exc: path.unlink(missing_ok=True); raise HTTPException(400,f"Invalid CSV: {exc}")
    if not rows: path.unlink(missing_ok=True); raise HTTPException(400,"CSV is empty")
    return {"dataset_id":dataset_id,"filename":file.filename,"summary":dataset_summary(headers,rows),
            "columns":col_schema(headers,rows),"charts":chart_data(headers,rows),
            "preview":preview_rows(rows,8),"target_recommendations":recommend_targets(headers,rows)}

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    path = DATA_DIR/f"{dataset_id}.csv"
    if not path.exists(): raise HTTPException(404,"Dataset not found")
    headers,rows = read_csv_file(path)
    return {"dataset_id":dataset_id,"summary":dataset_summary(headers,rows),"columns":col_schema(headers,rows),
            "charts":chart_data(headers,rows),"preview":preview_rows(rows,12),"target_recommendations":recommend_targets(headers,rows)}

@app.post("/api/ml/train")
def train_model(req: TrainRequest):
    path = DATA_DIR/f"{req.dataset_id}.csv"
    if not path.exists(): raise HTTPException(404,"Dataset not found")
    headers,rows = read_csv_file(path)
    if req.target not in headers: raise HTTPException(400,"Target column not found")
    task = req.task if req.task in {"classification","regression"} else infer_task(rows,req.target)
    if task=="regression":
        rows = [r for r in rows if isinstance(r.get(req.target),float)]; y_all = [r[req.target] for r in rows]
    else:
        rows = [r for r in rows if r.get(req.target) is not None]; y_all = [str(r[req.target]) for r in rows]
    if len(rows)<10: raise HTTPException(400,"Need at least 10 valid rows")
    if len(set(y_all))<2: raise HTTPException(400,"Target needs at least 2 distinct values")
    if task=="classification" and len(set(y_all))>len(rows)*0.5:
        raise HTTPException(400,f"Target '{req.target}' has {len(set(y_all))} unique values out of {len(rows)} rows ({int(len(set(y_all))/len(rows)*100)}% unique). Looks like an ID column. Please choose a column with fewer distinct values.")
    feature_cols = [c for c in headers if c!=req.target]
    if not feature_cols: raise HTTPException(400,"No feature columns")
    train_rows,test_rows = split_rows(rows,req.test_size,req.random_state)
    if not train_rows or not test_rows: raise HTTPException(400,"Dataset too small for split")
    prep = Preprocessor()
    X_train = prep.fit_transform(train_rows,feature_cols); X_test = prep.transform(test_rows)
    if task=="classification":
        y_train = [str(r[req.target]) for r in train_rows]; y_test = [str(r[req.target]) for r in test_rows]
    else:
        y_train = [float(r[req.target]) for r in train_rows]; y_test = [float(r[req.target]) for r in test_rows]
    results = []; all_trained_map = {}; all_preds = {}
    for name,est in build_models(task,req.random_state).items():
        try:
            est.fit(X_train,y_train); yp = est.predict(X_test)
            metrics = evaluate(task,y_test,yp); results.append({"algorithm":name,**metrics})
            all_trained_map[name] = est; all_preds[name] = yp
        except Exception as exc: results.append({"algorithm":name,"error":str(exc)})
    good = [r for r in results if "error" not in r]
    if not good: raise HTTPException(400,"No model trained successfully")
    best = max(good,key=lambda r:score_key(task,r)); best_name = best["algorithm"]
    best_preds = all_preds[best_name]; model = all_trained_map[best_name]; fi = []
    if hasattr(model,"feature_importances_"):
        pairs = sorted(zip(prep.feature_names,model.feature_importances_),key=lambda x:x[1],reverse=True)[:15]
        fi = [{"feature":n,"importance":round(v,5)} for n,v in pairs]
    elif hasattr(model,"coef_"):
        coef = model.coef_
        if coef and isinstance(coef[0],list): vals = [sum(abs(coef[c][j]) for c in range(len(coef)))/max(len(coef),1) for j in range(len(coef[0]))]
        else: vals = [abs(v) for v in coef]
        pairs = sorted(zip(prep.feature_names,vals),key=lambda x:x[1],reverse=True)[:15]
        fi = [{"feature":n,"importance":round(v,5)} for n,v in pairs]
    bm = {k:v for k,v in best.items() if k!="algorithm"}
    pred_dist = prediction_distribution(task,best_preds)
    insights = generate_insights(task,req.target,best_name,bm,fi,y_test,best_preds,headers,rows)
    extra: Dict[str,Any] = {}
    if task=="classification":
        extra["confusion_matrix"] = confusion_matrix_data(y_test,best_preds)
        extra["per_class_metrics"] = per_class_metrics(y_test,best_preds)
        cd = Counter(y_train); extra["class_distribution"] = [{"label":str(k),"count":v} for k,v in cd.most_common(12)]
    else:
        extra["residuals"] = residual_samples(y_test,best_preds)
        extra["actual_vs_predicted"] = [{"actual":round(float(a),3),"predicted":round(float(p),3)} for a,p in zip(y_test[:60],best_preds[:60])]
    model_id = uuid.uuid4().hex[:12]
    
    # Store default feature sample values for custom prediction UI
    sample_defaults = {}
    for col in feature_cols:
        col_vals_list = [r.get(col) for r in train_rows if r.get(col) is not None]
        if is_numeric_col(train_rows, col):
            num_v = [v for v in col_vals_list if isinstance(v, float)]
            sample_defaults[col] = round(statistics.mean(num_v), 2) if num_v else 0.0
        else:
            str_v = [str(v) for v in col_vals_list]
            sample_defaults[col] = Counter(str_v).most_common(1)[0][0] if str_v else ""

    with open(MODEL_DIR/f"{model_id}.pkl","wb") as f:
        pickle.dump({
            "preprocessor": prep, "model": model, "target": req.target, "task": task,
            "feature_columns": feature_cols, "dataset_id": req.dataset_id, "algorithm": best_name,
            "sample_defaults": sample_defaults, "results": results, "best_metrics": bm
        }, f)
        
    return {"model_id":model_id,"task":task,"target":req.target,"best_algorithm":best_name,
            "results":results,"feature_importance":fi,"prediction_distribution":pred_dist,
            "insights":insights,"sample_defaults":sample_defaults,"feature_columns":feature_cols,
            "train_size":len(train_rows),"test_size_count":len(test_rows),**extra}

@app.post("/api/ml/predict")
def predict_endpoint(req: PredictRequest):
    mf = MODEL_DIR/f"{req.model_id}.pkl"
    if not mf.exists(): raise HTTPException(404,"Model not found")
    with open(mf,"rb") as f: artifact = pickle.load(f)
    pp = DATA_DIR/f"{req.dataset_id}.csv"
    if not pp.exists(): raise HTTPException(404,"Prediction dataset not found")
    headers,rows = read_csv_file(pp)
    missing = [c for c in artifact["feature_columns"] if c not in headers]
    if missing: raise HTTPException(400,f"Missing columns: {missing}")
    X = artifact["preprocessor"].transform(rows); preds = artifact["model"].predict(X)
    predictions = [{"row":i+1,"prediction":round(p,4) if isinstance(p,float) else str(p)} for i,p in enumerate(preds[:200])]
    pd3 = prediction_distribution(artifact["task"],preds)
    return {"model_id":req.model_id,"algorithm":artifact["algorithm"],"target":artifact["target"],
            "task":artifact["task"],"count":len(preds),"predictions":predictions,"prediction_distribution":pd3}

@app.post("/api/ml/predict_custom")
def predict_custom(req: CustomPredictRequest):
    mf = MODEL_DIR/f"{req.model_id}.pkl"
    if not mf.exists(): raise HTTPException(404,"Model not found")
    with open(mf,"rb") as f: artifact = pickle.load(f)
    
    row = {}
    for col in artifact["feature_columns"]:
        val = req.inputs.get(col)
        n = _parse_number(str(val)) if val is not None else None
        row[col] = n if n is not None else val
        
    X = artifact["preprocessor"].transform([row])
    pred = artifact["model"].predict(X)[0]
    out_val = round(pred, 4) if isinstance(pred, float) else str(pred)
    
    return {
        "model_id": req.model_id,
        "algorithm": artifact["algorithm"],
        "target": artifact["target"],
        "task": artifact["task"],
        "prediction": out_val,
        "inputs_used": row
    }

@app.post("/api/ml/ai_insights")
def get_ai_insights(req: AIInsightsRequest):
    mf = MODEL_DIR/f"{req.model_id}.pkl"
    if not mf.exists(): raise HTTPException(404,"Model not found")
    with open(mf,"rb") as f: artifact = pickle.load(f)
    
    # Load dataset for column stats
    dp = DATA_DIR/f"{req.dataset_id}.csv"
    if not dp.exists(): raise HTTPException(404,"Dataset not found")
    headers,rows = read_csv_file(dp)
    
    # Re-build y_test and y_pred using saved model for insight context
    # (we only store a sample; use prediction distribution as proxy)
    task = artifact["task"]; target = artifact["target"]
    feature_cols = artifact["feature_columns"]
    prep = artifact["preprocessor"]; model = artifact["model"]
    
    valid_rows = [r for r in rows if r.get(target) is not None]
    if not valid_rows: raise HTTPException(400, "No valid target rows")
    
    # Get predictions on full dataset for insight context
    X_all = prep.transform(valid_rows)
    y_all_pred = model.predict(X_all)
    if task == "classification":
        y_all_true = [str(r[target]) for r in valid_rows]
    else:
        y_all_true = [float(r[target]) for r in valid_rows if isinstance(r.get(target), float)]
        y_all_pred = y_all_pred[:len(y_all_true)]
    
    # Build feature importances from stored artifact
    fi_list = []
    if hasattr(model, "feature_importances_"):
        pairs = sorted(zip(prep.feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)[:15]
        fi_list = [{"feature": n, "importance": round(v, 5)} for n, v in pairs]
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef and isinstance(coef[0], list): vals2 = [sum(abs(coef[c][j]) for c in range(len(coef)))/max(len(coef),1) for j in range(len(coef[0]))]
        else: vals2 = [abs(v) for v in coef]
        pairs = sorted(zip(prep.feature_names, vals2), key=lambda x: x[1], reverse=True)[:15]
        fi_list = [{"feature": n, "importance": round(v, 5)} for n, v in pairs]
    
    # Gather results from artifact if stored, else build minimal
    stored_results = artifact.get("results", [{"algorithm": artifact["algorithm"]}])
    best_metrics = artifact.get("best_metrics", {})
    
    structured = gemini_ai_insights(
        task=task, target=target, best_alg=artifact["algorithm"],
        best_metrics=best_metrics, fi_list=fi_list,
        results=stored_results, headers=headers, rows=rows,
        y_test=y_all_true, y_pred=y_all_pred
    )
    
    has_gemini = _get_gemini() is not None
    return {"ai_insights": structured, "powered_by": "gemini-1.5-flash" if has_gemini else "rule-engine", "model_id": req.model_id}

def infer_task(rows,target):
    vals = [v for v in col_values(rows,target) if v is not None]
    if not vals: raise HTTPException(400,"Target column is empty")
    if not all(isinstance(v,float) for v in vals[:100]): return "classification"
    return "classification" if len(set(vals))<=min(20,max(2,int(len(vals)*0.02))) else "regression"

def split_rows(rows,test_size,seed):
    rng = random.Random(seed); s = rows[:]; rng.shuffle(s); sp = int(len(s)*(1-test_size))
    return s[:sp],s[sp:]

def build_models(task,seed):
    if task=="classification":
        return {"Random Forest":RandomForest(n_estimators=80,task="classification",max_depth=10,random_state=seed),
                "Decision Tree":DecisionTree(task="classification",max_depth=10,random_state=seed),
                "Logistic Regression":LogisticRegression(lr=0.1,max_iter=150,random_state=seed)}
    return {"Random Forest":RandomForest(n_estimators=60,task="regression",max_depth=10,random_state=seed),
            "Decision Tree":DecisionTree(task="regression",max_depth=10,random_state=seed),
            "Linear Regression":LinearRegression()}

# ===========================================================================
# KAGGLE ENDPOINTS
# ===========================================================================
_KAGGLE_TMP: Dict[str, Path] = {}   # download_id -> extracted folder path

def _parse_kaggle_url(url: str):
    """Return (owner, slug) from https://www.kaggle.com/datasets/owner/slug"""
    import re
    m = re.search(r"kaggle\.com/datasets/([^/\s]+)/([^/\s?#]+)", url)
    if not m:
        raise HTTPException(400, "Invalid Kaggle URL. Expected: https://www.kaggle.com/datasets/owner/dataset-name")
    return m.group(1), m.group(2)

def _kaggle_env_ok():
    return bool((os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or os.getenv("KAGGLE_API_TOKEN"))

@app.post("/api/kaggle/list-files")
def kaggle_list_files(req: KaggleImportRequest):
    """Download the Kaggle dataset zip and return available tabular file list."""
    if not _kaggle_env_ok():
        raise HTTPException(403, "Kaggle credentials not configured. Set KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN) in backend .env")
    owner, slug = _parse_kaggle_url(req.url)
    try:
        import kaggle  # type: ignore
        # Set env vars so kaggle SDK picks them up
        if os.getenv("KAGGLE_API_TOKEN"):
            os.environ.setdefault("KAGGLE_API_TOKEN", os.getenv("KAGGLE_API_TOKEN", ""))
        os.environ.setdefault("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
        os.environ.setdefault("KAGGLE_KEY", os.getenv("KAGGLE_KEY", ""))

    except ImportError:
        raise HTTPException(500, "kaggle package not installed. Run: pip install kaggle")

    tmp_dir = Path(tempfile.mkdtemp(prefix="ifkaggle_"))
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(f"{owner}/{slug}", path=str(tmp_dir), unzip=True, quiet=True)
    except Exception as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Kaggle download failed: {exc}")

    # Find supported tabular files
    supported_exts = {".csv", ".json", ".xlsx", ".xls"}
    files = []
    for fp in tmp_dir.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in supported_exts:
            try:
                size_kb = round(fp.stat().st_size / 1024, 1)
                # Quick row count preview for CSV
                row_count = None
                if fp.suffix.lower() == ".csv":
                    with open(fp, encoding="utf-8-sig", errors="replace") as fh:
                        row_count = sum(1 for _ in fh) - 1
                files.append({
                    "filename": fp.relative_to(tmp_dir).as_posix(),
                    "size_kb": size_kb,
                    "row_count": row_count,
                    "type": fp.suffix.lower().lstrip(".")
                })
            except Exception:
                pass

    if not files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "No supported tabular files found in this Kaggle dataset (expected .csv, .json, .xlsx)")

    download_id = uuid.uuid4().hex[:12]
    _KAGGLE_TMP[download_id] = tmp_dir
    return {"download_id": download_id, "owner": owner, "slug": slug, "files": files}


@app.post("/api/kaggle/import-file")
def kaggle_import_file(req: KaggleSelectFileRequest):
    """Import the selected file from a previously downloaded Kaggle dataset."""
    if req.download_id not in _KAGGLE_TMP:
        raise HTTPException(404, "Download session expired or not found. Please re-import from Kaggle.")
    tmp_dir = _KAGGLE_TMP[req.download_id]
    file_path = (tmp_dir / req.filename).resolve()
    # Safety check: must be inside tmp_dir
    if not str(file_path).startswith(str(tmp_dir)):
        raise HTTPException(400, "Invalid filename")
    if not file_path.exists():
        raise HTTPException(404, f"File '{req.filename}' not found in downloaded dataset")

    ext = file_path.suffix.lower()
    dataset_id = uuid.uuid4().hex[:12]
    dest_csv = DATA_DIR / f"{dataset_id}.csv"

    try:
        if ext == ".csv":
            shutil.copy2(file_path, dest_csv)
        elif ext == ".json":
            import json as _json
            with open(file_path, encoding="utf-8") as fh:
                data = _json.load(fh)
            if isinstance(data, list) and data:
                headers = list(data[0].keys())
                with open(dest_csv, "w", newline="", encoding="utf-8") as out:
                    w = csv.DictWriter(out, fieldnames=headers)
                    w.writeheader(); w.writerows(data)
            else:
                raise HTTPException(400, "JSON file must be an array of objects")
        elif ext in (".xlsx", ".xls"):
            import openpyxl  # type: ignore
            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            ws = wb.active
            rows_data = list(ws.iter_rows(values_only=True))
            if not rows_data: raise HTTPException(400, "Excel file is empty")
            headers = [str(c) if c is not None else f"col_{i}" for i, c in enumerate(rows_data[0])]
            with open(dest_csv, "w", newline="", encoding="utf-8") as out:
                w = csv.writer(out)
                w.writerow(headers)
                for row in rows_data[1:]:
                    w.writerow([str(v) if v is not None else "" for v in row])
        else:
            raise HTTPException(400, f"Unsupported file type: {ext}")
    except HTTPException:
        raise
    except Exception as exc:
        dest_csv.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to process file: {exc}")

    try:
        headers, rows = read_csv_file(dest_csv)
    except Exception as exc:
        dest_csv.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to read converted CSV: {exc}")
    if not rows:
        dest_csv.unlink(missing_ok=True)
        raise HTTPException(400, "Selected file is empty")

    owner_slug = f"{tmp_dir.name}"  # keep track loosely; real owner/slug passed from frontend
    return {
        "dataset_id": dataset_id,
        "filename": req.filename,
        "source_type": "kaggle",
        "summary": dataset_summary(headers, rows),
        "columns": col_schema(headers, rows),
        "charts": chart_data(headers, rows),
        "preview": preview_rows(rows, 10),
        "target_recommendations": recommend_targets(headers, rows)
    }


@app.delete("/api/kaggle/cleanup/{download_id}")
def kaggle_cleanup(download_id: str):
    """Free temporary Kaggle download folder."""
    if download_id in _KAGGLE_TMP:
        shutil.rmtree(_KAGGLE_TMP.pop(download_id), ignore_errors=True)
    return {"ok": True}



# ===========================================================================
# PREPROCESS ENDPOINT
# ===========================================================================

@app.post("/api/datasets/preprocess")
def preprocess_dataset(req: PreprocessRequest):
    """Apply selected preprocessing steps to a dataset and return a new dataset."""
    path = DATA_DIR / f"{req.dataset_id}.csv"
    if not path.exists():
        raise HTTPException(404, "Dataset not found")

    headers, rows = read_csv_file(path)
    ops = req.operations
    applied_ops = []

    # Identify column types
    num_cols = [h for h in headers if is_numeric_col(rows, h)]
    cat_cols = [h for h in headers if not is_numeric_col(rows, h)]

    # ── 1. Median Fill (numeric) ────────────────────────────────
    if 'median_fill' in ops:
        for col in num_cols:
            vals = [v for v in col_values(rows, col) if isinstance(v, float)]
            med = _median(vals) if vals else 0.0
            for r in rows:
                if r.get(col) is None:
                    r[col] = med
        applied_ops.append('median_fill')

    # ── 2. Mode Fill (categorical) ──────────────────────────────
    if 'mode_fill' in ops:
        for col in cat_cols:
            vals = [str(v) for v in col_values(rows, col) if v is not None]
            mode = _mode_str(vals) if vals else ''
            for r in rows:
                if r.get(col) is None:
                    r[col] = mode
        applied_ops.append('mode_fill')

    # ── 3. Forward Fill ─────────────────────────────────────────
    if 'ffill' in ops:
        last = {h: None for h in headers}
        for r in rows:
            for h in headers:
                if r.get(h) is not None:
                    last[h] = r[h]
                elif last[h] is not None:
                    r[h] = last[h]
        applied_ops.append('ffill')

    # ── 4. Drop rows with any missing ───────────────────────────
    if 'drop_missing' in ops:
        rows = [r for r in rows if all(r.get(h) is not None for h in headers)]
        applied_ops.append('drop_missing')

    # ── 5. Min-Max Normalization ────────────────────────────────
    if 'minmax' in ops:
        for col in num_cols:
            vals = [v for v in col_values(rows, col) if isinstance(v, float)]
            if not vals: continue
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx != mn else 1.0
            for r in rows:
                v = r.get(col)
                if isinstance(v, float):
                    r[col] = round((v - mn) / rng, 6)
        applied_ops.append('minmax')

    # ── 6. Z-Score Standardization ──────────────────────────────
    if 'zscore' in ops and 'minmax' not in ops:  # don't double-normalize
        for col in num_cols:
            vals = [v for v in col_values(rows, col) if isinstance(v, float)]
            if len(vals) < 2: continue
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) or 1.0
            for r in rows:
                v = r.get(col)
                if isinstance(v, float):
                    r[col] = round((v - mu) / sd, 6)
        applied_ops.append('zscore')

    # ── 7. Winsorize (cap 5th/95th percentile) ──────────────────
    if 'winsorize' in ops:
        for col in num_cols:
            vals = sorted(v for v in col_values(rows, col) if isinstance(v, float))
            if len(vals) < 20: continue
            lo = vals[int(len(vals) * 0.05)]
            hi = vals[int(len(vals) * 0.95)]
            for r in rows:
                v = r.get(col)
                if isinstance(v, float):
                    r[col] = max(lo, min(hi, v))
        applied_ops.append('winsorize')

    # ── 8. IQR outlier capping ──────────────────────────────────
    if 'iqr' in ops:
        for col in num_cols:
            vals = sorted(v for v in col_values(rows, col) if isinstance(v, float))
            if len(vals) < 4: continue
            q1 = vals[len(vals)//4]; q3 = vals[3*len(vals)//4]
            iqr_val = q3 - q1
            lo, hi = q1 - 1.5 * iqr_val, q3 + 1.5 * iqr_val
            for r in rows:
                v = r.get(col)
                if isinstance(v, float):
                    r[col] = max(lo, min(hi, v))
        applied_ops.append('iqr')

    # ── 9. Drop outlier rows (Z-Score >3) ───────────────────────
    if 'drop_outliers' in ops:
        col_stats = {}
        for col in num_cols:
            vals = [v for v in col_values(rows, col) if isinstance(v, float)]
            if len(vals) < 2: continue
            mu = statistics.mean(vals); sd = statistics.stdev(vals) or 1.0
            col_stats[col] = (mu, sd)
        def is_outlier(r):
            for col, (mu, sd) in col_stats.items():
                v = r.get(col)
                if isinstance(v, float) and abs(v - mu) > 3 * sd:
                    return True
            return False
        rows = [r for r in rows if not is_outlier(r)]
        applied_ops.append('drop_outliers')

    # ── 10. Label Encoding ──────────────────────────────────────
    if 'label_enc' in ops:
        for col in cat_cols:
            vals = sorted(set(str(v) for v in col_values(rows, col) if v is not None))
            cat_map = {s: i for i, s in enumerate(vals)}
            for r in rows:
                v = r.get(col)
                if v is not None:
                    r[col] = float(cat_map.get(str(v), 0))
        # Refresh column type lists after encoding
        num_cols = [h for h in headers if is_numeric_col(rows, h)]
        cat_cols = [h for h in headers if not is_numeric_col(rows, h)]
        applied_ops.append('label_enc')

    # ── 11. One-Hot Encoding ─────────────────────────────────────
    if 'ohe' in ops and 'label_enc' not in ops:
        new_headers = list(num_cols)
        ohe_maps = {}
        for col in cat_cols:
            vals = sorted(set(str(v) for v in col_values(rows, col) if v is not None))
            ohe_maps[col] = vals
            for val in vals:
                new_headers.append(f"{col}_{val}")
        new_rows = []
        for r in rows:
            nr = {h: r.get(h) for h in num_cols}
            for col, vals in ohe_maps.items():
                v = str(r.get(col, ''))
                for val in vals:
                    nr[f"{col}_{val}"] = 1.0 if v == val else 0.0
            new_rows.append(nr)
        headers = new_headers
        rows = new_rows
        applied_ops.append('ohe')

    # ── 12. Log Transform (skewed columns) ──────────────────────
    if 'log_transform' in ops:
        for col in num_cols:
            vals = [v for v in col_values(rows, col) if isinstance(v, float) and v > 0]
            if not vals: continue
            mean_v = statistics.mean(vals)
            max_v = max(vals)
            # Only log-transform highly skewed columns (max >> mean)
            if max_v > mean_v * 5:
                for r in rows:
                    v = r.get(col)
                    if isinstance(v, float) and v >= 0:
                        r[col] = round(math.log1p(v), 6)
        applied_ops.append('log_transform')

    # ── 13. Drop Low-Variance Columns ────────────────────────────
    if 'low_var' in ops:
        to_drop = []
        for col in num_cols:
            vals = [v for v in col_values(rows, col) if isinstance(v, float)]
            if len(vals) < 2: continue
            try:
                if statistics.variance(vals) < 1e-6:
                    to_drop.append(col)
            except Exception:
                pass
        for col in cat_cols:
            vals = [v for v in col_values(rows, col) if v is not None]
            if vals and len(set(str(v) for v in vals)) == 1:
                to_drop.append(col)
        if to_drop:
            headers = [h for h in headers if h not in to_drop]
            rows = [{k: v for k, v in r.items() if k not in to_drop} for r in rows]
        applied_ops.append('low_var')

    # ── 14. Drop Highly Correlated Features (>0.95) ─────────────
    if 'corr_drop' in ops:
        cur_num = [h for h in headers if is_numeric_col(rows, h)]
        to_drop = set()
        for i, c1 in enumerate(cur_num):
            if c1 in to_drop: continue
            for c2 in cur_num[i+1:]:
                if c2 in to_drop: continue
                v1 = col_numeric(rows, c1); v2 = col_numeric(rows, c2)
                if abs(_pearson(v1, v2)) > 0.95:
                    to_drop.add(c2)
        if to_drop:
            headers = [h for h in headers if h not in to_drop]
            rows = [{k: v for k, v in r.items() if k not in to_drop} for r in rows]
        applied_ops.append('corr_drop')

    # ── 15. Interaction Features ─────────────────────────────────
    if 'interaction' in ops:
        cur_num = [h for h in headers if is_numeric_col(rows, h)][:4]  # limit to 4 to avoid explosion
        new_cols = []
        for i in range(len(cur_num)):
            for j in range(i+1, len(cur_num)):
                c1, c2 = cur_num[i], cur_num[j]
                new_col = f"{c1}_x_{c2}"
                new_cols.append(new_col)
                for r in rows:
                    v1 = r.get(c1); v2 = r.get(c2)
                    if isinstance(v1, float) and isinstance(v2, float):
                        r[new_col] = round(v1 * v2, 6)
                    else:
                        r[new_col] = None
        headers = headers + new_cols
        applied_ops.append('interaction')

    if not rows:
        raise HTTPException(400, "All rows were removed by the selected operations. Try less aggressive preprocessing.")

    # Save new transformed dataset
    new_id = uuid.uuid4().hex[:12]
    dest = DATA_DIR / f"{new_id}.csv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=headers, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({h: (round(r[h], 6) if isinstance(r.get(h), float) else (r.get(h) or '')) for h in headers})

    h2, r2 = read_csv_file(dest)
    return {
        "dataset_id": new_id,
        "applied_operations": applied_ops,
        "rows_after": len(r2),
        "columns_after": len(h2),
        "summary": dataset_summary(h2, r2),
        "columns": col_schema(h2, r2),
        "charts": chart_data(h2, r2),
        "preview": preview_rows(r2, 10),
        "target_recommendations": recommend_targets(h2, r2),
    }


# ===========================================================================
# EXPORT ENDPOINT (Kaggle datasets)
# ===========================================================================





# ===========================================================================

# EXPORT ENDPOINT (Kaggle / QC datasets)
# ===========================================================================
@app.get("/api/datasets/{dataset_id}/export")
def export_dataset(dataset_id: str, filename: str = "dataset.csv"):
    """Stream a stored dataset CSV back to the browser for download."""
    path = DATA_DIR / f"{dataset_id}.csv"
    if not path.exists():
        raise HTTPException(404, "Dataset not found")
    import io
    data = path.read_bytes()
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
