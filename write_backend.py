
import os

target = os.path.join(os.path.dirname(__file__), "backend", "app", "main.py")

content = r'''"""
InsightForge Analytics - Pure-Python FastAPI backend v2.1
Enhanced: confusion matrix, per-class metrics, prediction distribution,
residuals, AI insights, target column recommendations, ID-column guard.
"""
from __future__ import annotations
import csv, math, pickle, random, statistics, uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "uploads"
MODEL_DIR = BASE / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

app = FastAPI(title="InsightForge Analytics API", version="2.1.0")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class TrainRequest(BaseModel):
    dataset_id: str; target: str; task: str
    test_size: float = 0.2; random_state: int = 42

class PredictRequest(BaseModel):
    model_id: str; dataset_id: str

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
# CART Decision Tree
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
        cnt = Counter(y); node.value=cnt.most_common(1)[0][0]; node.class_counts=dict(cnt); cv=_gini(y)
    else:
        node.value = statistics.mean(y) if y else 0.0; cv = _mse_var(y) if y else 0.0
    if depth>=max_depth or len(y)<min_split or len(set(y))==1 or cv==0.0: return node
    n_f = len(X[0]) if X else 0; fi = rng.sample(range(n_f), min(max_feats,n_f))
    bf,bt,_ = _best_split(X,y,fi,task,min_leaf)
    if bf==-1: return node
    lm = [X[i][bf]<=bt for i in range(len(y))]
    Xl=[X[i] for i in range(len(y)) if lm[i]]; yl=[y[i] for i in range(len(y)) if lm[i]]
    Xr=[X[i] for i in range(len(y)) if not lm[i]]; yr=[y[i] for i in range(len(y)) if not lm[i]]
    if not Xl or not Xr: return node
    node.feat,node.thresh = bf,bt
    node.left = _build_tree(Xl,yl,task,max_depth,min_split,min_leaf,max_feats,depth+1,rng)
    node.right = _build_tree(Xr,yr,task,max_depth,min_split,min_leaf,max_feats,depth+1,rng)
    return node

def _predict_node(node, x):
    if node.feat is None: return node.value
    return _predict_node(node.left if x[node.feat]<=node.thresh else node.right, x)

def _count_splits(node, counts):
    if node is None or node.feat is None: return
    counts[node.feat] += node.n_samples
    _count_splits(node.left,counts); _count_splits(node.right,counts)

class DecisionTree:
    def __init__(self,task="classification",max_depth=10,min_samples_split=2,min_samples_leaf=1,max_features=None,random_state=42):
        self.task=task; self.max_depth=max_depth; self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf; self.max_features=max_features; self.random_state=random_state
        self.root=None; self.feature_importances_=[]
    def fit(self,X,y):
        rng=random.Random(self.random_state); mf=self.max_features or (len(X[0]) if X else 1)
        self.root=_build_tree(X,y,self.task,self.max_depth,self.min_samples_split,self.min_samples_leaf,mf,0,rng)
        counts=[0]*(len(X[0]) if X else 1); _count_splits(self.root,counts)
        total=sum(counts) or 1; self.feature_importances_=[c/total for c in counts]
    def predict(self,X): return [_predict_node(self.root,x) for x in X]

# ===========================================================================
# Random Forest
# ===========================================================================
class RandomForest:
    def __init__(self,n_estimators=80,task="classification",max_depth=10,min_samples_split=2,random_state=42):
        self.n_estimators=n_estimators; self.task=task; self.max_depth=max_depth
        self.min_samples_split=min_samples_split; self.random_state=random_state
        self.trees=[]; self.feature_importances_=[]
    def fit(self,X,y):
        rng=random.Random(self.random_state); n,n_f=len(X),(len(X[0]) if X else 1)
        mf=max(1,int(math.sqrt(n_f))) if self.task=="classification" else max(1,n_f//3)
        imps=[0.0]*n_f; self.trees=[]
        for _ in range(self.n_estimators):
            idx=[rng.randint(0,n-1) for _ in range(n)]
            Xb,yb=[X[i] for i in idx],[y[i] for i in idx]
            tree=DecisionTree(task=self.task,max_depth=self.max_depth,min_samples_split=self.min_samples_split,max_features=mf,random_state=rng.randint(0,2**31))
            tree.fit(Xb,yb); self.trees.append(tree)
            for j,v in enumerate(tree.feature_importances_): imps[j]+=v
        total=sum(imps) or 1; self.feature_importances_=[v/total for v in imps]
    def predict(self,X):
        all_p=[tree.predict(X) for tree in self.trees]; out=[]
        for i in range(len(X)):
            votes=[all_p[t][i] for t in range(len(self.trees))]
            if self.task=="classification": out.append(Counter(votes).most_common(1)[0][0])
            else: out.append(statistics.mean(v for v in votes if isinstance(v,(int,float))))
        return out

# ===========================================================================
# Linear Regression (OLS)
# ===========================================================================
def _dot(a,b): return sum(x*y for x,y in zip(a,b))
def _matmul(A,B):
    rA,cA,cB=len(A),len(A[0]),len(B[0]); C=[[0.0]*cB for _ in range(rA)]
    for i in range(rA):
        for k in range(cA):
            if A[i][k]==0: continue
            for j in range(cB): C[i][j]+=A[i][k]*B[k][j]
    return C
def _T(A): return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]
def _solve(A,b):
    n=len(A); M=[row[:]+[b[i]] for i,row in enumerate(A)]
    for col in range(n):
        mr=max(range(col,n),key=lambda r:abs(M[r][col])); M[col],M[mr]=M[mr],M[col]; p=M[col][col]
        if abs(p)<1e-12: continue
        for row in range(col+1,n):
            f=M[row][col]/p
            for j in range(col,n+1): M[row][j]-=f*M[col][j]
    x=[0.0]*n
    for i in range(n-1,-1,-1):
        if abs(M[i][i])<1e-12: continue
        x[i]=(M[i][n]-sum(M[i][j]*x[j] for j in range(i+1,n)))/M[i][i]
    return x

class LinearRegression:
    def __init__(self): self.coef_=[]; self.intercept_=0.0
    def fit(self,X,y):
        Xb=[[1.0]+row for row in X]; Xt=_T(Xb); XtX=_matmul(Xt,Xb)
        for i in range(len(XtX)): XtX[i][i]+=1e-4
        Xty=[_dot(Xt[i],y) for i in range(len(Xt))]; w=_solve(XtX,Xty)
        self.intercept_,self.coef_=w[0],w[1:]
    def predict(self,X): return [self.intercept_+_dot(self.coef_,x) for x in X]

# ===========================================================================
# Logistic Regression (gradient descent + OvR)
# ===========================================================================
def _sig(z):
    if z<-500: return 0.0
    if z>500: return 1.0
    return 1.0/(1.0+math.exp(-z))

class LogisticRegression:
    def __init__(self,lr=0.1,max_iter=150,random_state=42):
        self.lr=lr; self.max_iter=max_iter; self.random_state=random_state
        self.classes_=[]; self.weights_=[]; self.coef_=[]
    def _fit_binary(self,X,y_bin):
        n,p=len(X),len(X[0]); rng=random.Random(self.random_state)
        w=[rng.gauss(0,0.01) for _ in range(p+1)]
        for _ in range(self.max_iter):
            g=[0.0]*(p+1)
            for i in range(n):
                err=_sig(w[0]+_dot(w[1:],X[i]))-y_bin[i]; g[0]+=err
                for j in range(p): g[j+1]+=err*X[i][j]
            for j in range(p+1): w[j]-=self.lr*g[j]/n
        return w
    def fit(self,X,y):
        self.classes_=sorted(set(y))
        self.weights_=[self._fit_binary(X,[1 if yi==c else 0 for yi in y]) for c in self.classes_]
        self.coef_=[w[1:] for w in self.weights_]
    def predict(self,X):
        out=[]
        for x in X:
            scores=[_sig(w[0]+_dot(w[1:],x)) for w in self.weights_]
            out.append(self.classes_[scores.index(max(scores))])
        return out

# ===========================================================================
# Metrics
# ===========================================================================
def accuracy_score(yt,yp): return sum(a==b for a,b in zip(yt,yp))/len(yt) if yt else 0.0

def precision_recall_f1_weighted(yt,yp):
    classes=set(yt); total=len(yt); counts=Counter(yt); ps=rs=fs=0.0
    for cls in classes:
        tp=sum(1 for a,b in zip(yt,yp) if a==cls and b==cls)
        fp=sum(1 for a,b in zip(yt,yp) if a!=cls and b==cls)
        fn=sum(1 for a,b in zip(yt,yp) if a==cls and b!=cls)
        p=tp/(tp+fp) if (tp+fp) else 0.0; r=tp/(tp+fn) if (tp+fn) else 0.0
        f=2*p*r/(p+r) if (p+r) else 0.0; w=counts[cls]/total
        ps+=p*w; rs+=r*w; fs+=f*w
    return round(ps,4),round(rs,4),round(fs,4)

def per_class_metrics(yt,yp):
    classes=sorted(set(yt)); results=[]
    for cls in classes:
        tp=sum(1 for a,b in zip(yt,yp) if a==cls and b==cls)
        fp=sum(1 for a,b in zip(yt,yp) if a!=cls and b==cls)
        fn=sum(1 for a,b in zip(yt,yp) if a==cls and b!=cls)
        support=sum(1 for a in yt if a==cls)
        p=round(tp/(tp+fp),4) if (tp+fp) else 0.0
        r=round(tp/(tp+fn),4) if (tp+fn) else 0.0
        f=round(2*p*r/(p+r),4) if (p+r) else 0.0
        results.append({"class":str(cls),"precision":p,"recall":r,"f1":f,"support":support})
    return results

def confusion_matrix_data(yt,yp):
    top_classes=[c for c,_ in Counter(yt).most_common(8)]; classes=sorted(top_classes)
    idx={c:i for i,c in enumerate(classes)}; n=len(classes)
    mat=[[0]*n for _ in range(n)]
    for a,b in zip(yt,yp):
        if a in idx and b in idx: mat[idx[a]][idx[b]]+=1
    return {"classes":[str(c) for c in classes],"matrix":mat}

def r2_score(yt,yp):
    my=statistics.mean(yt); ss_res=sum((a-b)**2 for a,b in zip(yt,yp)); ss_tot=sum((a-my)**2 for a in yt)
    return round(1-ss_res/ss_tot,4) if ss_tot else 0.0

def mae(yt,yp): return round(statistics.mean(abs(a-b) for a,b in zip(yt,yp)),4)
def rmse(yt,yp): return round(math.sqrt(statistics.mean((a-b)**2 for a,b in zip(yt,yp))),4)

def evaluate(task,yt,yp):
    if task=="classification":
        p,r,f=precision_recall_f1_weighted(yt,yp)
        return {"accuracy":round(accuracy_score(yt,yp),4),"precision":p,"recall":r,"f1":f}
    yt2,yp2=[float(v) for v in yt],[float(v) for v in yp]
    return {"r2":r2_score(yt2,yp2),"mae":mae(yt2,yp2),"rmse":rmse(yt2,yp2)}

def score_key(task,metrics): return metrics.get("accuracy",metrics.get("r2",-999))

def prediction_distribution(task,yp):
    if task=="classification":
        counts=Counter(str(v) for v in yp); total=len(yp)
        return [{"label":k,"count":v,"pct":round(v/total*100,1)} for k,v in counts.most_common(12)]
    fv=[float(v) for v in yp if isinstance(v,(int,float))]
    if not fv: return []
    mn,mx=min(fv),max(fv); rng=mx-mn
    if rng==0: return [{"label":str(round(mn,2)),"count":len(fv),"pct":100.0}]
    bins=10; bin_size=rng/bins; buckets={}
    for v in fv:
        b=min(int((v-mn)/bin_size),bins-1); buckets[b]=buckets.get(b,0)+1
    total=len(fv)
    return [{"label":f"{round(mn+i*bin_size,1)}-{round(mn+(i+1)*bin_size,1)}","count":buckets.get(i,0),"pct":round(buckets.get(i,0)/total*100,1)} for i in range(bins)]

def residual_samples(yt,yp,n=60):
    pairs=list(zip([float(v) for v in yt],[float(v) for v in yp])); step=max(1,len(pairs)//n)
    return [{"actual":round(a,3),"predicted":round(p,3),"residual":round(a-p,3)} for a,p in pairs[::step][:n]]

def generate_insights(task,target,best_alg,best_metrics,fi_list,y_test,y_pred):
    insights=[]
    if task=="classification":
        acc=best_metrics.get("accuracy",0); f1=best_metrics.get("f1",0)
        if acc>=0.90: insights.append({"type":"success","title":f"Excellent Accuracy: {acc*100:.1f}%","text":f"{best_alg} achieved {acc*100:.1f}% accuracy on held-out test data. Production-ready model."})
        elif acc>=0.75: insights.append({"type":"good","title":f"Good Accuracy: {acc*100:.1f}%","text":f"Solid result. Consider feature engineering or more data to push accuracy higher."})
        elif acc>=0.60: insights.append({"type":"warn","title":f"Moderate Accuracy: {acc*100:.1f}%","text":f"Model is learning but struggles on harder cases. Try feature engineering or balancing data."})
        else: insights.append({"type":"error","title":f"Low Accuracy: {acc*100:.1f}%","text":f"Target '{target}' may be too complex, have too many classes, or need better feature representation."})
        if f1 < acc-0.10: insights.append({"type":"warn","title":"Class Imbalance Detected","text":f"F1 ({f1*100:.1f}%) is much lower than accuracy ({acc*100:.1f}%). Model favours majority classes. Try oversampling."})
        pd2=Counter(str(v) for v in y_pred)
        if len(pd2)==1: insights.append({"type":"error","title":"Single-Class Predictions","text":"All predictions are the same class. Severe imbalance - rebalance the dataset and retrain."})
    else:
        r2=best_metrics.get("r2",0); rmse_val=best_metrics.get("rmse",0); mae_val=best_metrics.get("mae",0)
        y_vals=[float(v) for v in y_test]; tm=statistics.mean(y_vals) if y_vals else 1.0
        cv_pct=rmse_val/abs(tm)*100 if tm else 0
        if r2>=0.85: insights.append({"type":"success","title":f"Strong R2 = {r2:.3f}","text":f"{best_alg} explains {r2*100:.1f}% of variance in '{target}'. RMSE={rmse_val:.2f}, target mean={tm:.2f}, CV={cv_pct:.1f}%."})
        elif r2>=0.65: insights.append({"type":"good","title":f"Good R2 = {r2:.3f}","text":f"Explains {r2*100:.1f}% of variance. MAE={mae_val:.2f}. Some non-linearity remains."})
        elif r2>=0.30: insights.append({"type":"warn","title":f"Weak R2 = {r2:.3f}","text":f"Only {r2*100:.1f}% variance explained. Target may be noisy or need more predictive features."})
        else: insights.append({"type":"error","title":f"Very Low R2 = {r2:.3f}","text":f"R2 near zero - model barely predicts '{target}'. Check data distribution and feature relevance."})
        rel_err=mae_val/abs(tm)*100 if tm else 0
        insights.append({"type":"info","title":f"Mean Absolute Error: {mae_val:.3f}","text":f"Average prediction error is {mae_val:.3f} units. Target mean={tm:.3f}, relative MAE={rel_err:.1f}%."})
    if fi_list and len(fi_list)>=2:
        t=fi_list[:2]; insights.append({"type":"info","title":f"Key Drivers: {t[0]['feature']} & {t[1]['feature']}","text":f"Top predictors: '{t[0]['feature']}' (importance={t[0]['importance']:.4f}), '{t[1]['feature']}' ({t[1]['importance']:.4f})."})
    elif fi_list:
        insights.append({"type":"info","title":f"Top Feature: {fi_list[0]['feature']}","text":f"'{fi_list[0]['feature']}' (importance={fi_list[0]['importance']:.4f}) is the dominant predictor for '{target}'."})
    return insights

# ===========================================================================
# Training helpers
# ===========================================================================
def infer_task(rows,target):
    vals=[v for v in col_values(rows,target) if v is not None]
    if not vals: raise HTTPException(400,"Target column is empty")
    if not all(isinstance(v,float) for v in vals[:100]): return "classification"
    return "classification" if len(set(vals))<=min(20,max(2,int(len(vals)*0.02))) else "regression"

def split_rows(rows,test_size,seed):
    rng=random.Random(seed); s=rows[:]; rng.shuffle(s); sp=int(len(s)*(1-test_size))
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
# API Endpoints
# ===========================================================================
@app.get("/api/health")
def health(): return {"status":"ok","service":"InsightForge Analytics API","engine":"pure-python"}

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Only CSV files are supported")
    raw=await file.read()
    if len(raw)>50*1024*1024: raise HTTPException(413,"CSV exceeds 50 MB")
    dataset_id=uuid.uuid4().hex[:12]; path=DATA_DIR/f"{dataset_id}.csv"; path.write_bytes(raw)
    try: headers,rows=read_csv_file(path)
    except Exception as exc: path.unlink(missing_ok=True); raise HTTPException(400,f"Invalid CSV: {exc}")
    if not rows: path.unlink(missing_ok=True); raise HTTPException(400,"CSV is empty")
    return {"dataset_id":dataset_id,"filename":file.filename,"summary":dataset_summary(headers,rows),
            "columns":col_schema(headers,rows),"charts":chart_data(headers,rows),
            "preview":preview_rows(rows,8),"target_recommendations":recommend_targets(headers,rows)}

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    path=DATA_DIR/f"{dataset_id}.csv"
    if not path.exists(): raise HTTPException(404,"Dataset not found")
    headers,rows=read_csv_file(path)
    return {"dataset_id":dataset_id,"summary":dataset_summary(headers,rows),"columns":col_schema(headers,rows),
            "charts":chart_data(headers,rows),"preview":preview_rows(rows,12),"target_recommendations":recommend_targets(headers,rows)}

@app.post("/api/ml/train")
def train_model(req: TrainRequest):
    path=DATA_DIR/f"{req.dataset_id}.csv"
    if not path.exists(): raise HTTPException(404,"Dataset not found")
    headers,rows=read_csv_file(path)
    if req.target not in headers: raise HTTPException(400,"Target column not found")
    task=req.task if req.task in {"classification","regression"} else infer_task(rows,req.target)
    if task=="regression":
        rows=[r for r in rows if isinstance(r.get(req.target),float)]; y_all=[r[req.target] for r in rows]
    else:
        rows=[r for r in rows if r.get(req.target) is not None]; y_all=[str(r[req.target]) for r in rows]
    if len(rows)<10: raise HTTPException(400,"Need at least 10 valid rows")
    if len(set(y_all))<2: raise HTTPException(400,"Target needs at least 2 distinct values")
    if task=="classification" and len(set(y_all))>len(rows)*0.5:
        raise HTTPException(400,f"Target '{req.target}' has {len(set(y_all))} unique values out of {len(rows)} rows ({int(len(set(y_all))/len(rows)*100)}% unique). Looks like an ID column. Please choose a column with fewer distinct values.")
    feature_cols=[c for c in headers if c!=req.target]
    if not feature_cols: raise HTTPException(400,"No feature columns")
    train_rows,test_rows=split_rows(rows,req.test_size,req.random_state)
    if not train_rows or not test_rows: raise HTTPException(400,"Dataset too small for split")
    prep=Preprocessor()
    X_train=prep.fit_transform(train_rows,feature_cols); X_test=prep.transform(test_rows)
    if task=="classification":
        y_train=[str(r[req.target]) for r in train_rows]; y_test=[str(r[req.target]) for r in test_rows]
    else:
        y_train=[float(r[req.target]) for r in train_rows]; y_test=[float(r[req.target]) for r in test_rows]
    results=[]; all_trained_map={}; all_preds={}
    for name,est in build_models(task,req.random_state).items():
        try:
            est.fit(X_train,y_train); yp=est.predict(X_test)
            metrics=evaluate(task,y_test,yp); results.append({"algorithm":name,**metrics})
            all_trained_map[name]=est; all_preds[name]=yp
        except Exception as exc: results.append({"algorithm":name,"error":str(exc)})
    good=[r for r in results if "error" not in r]
    if not good: raise HTTPException(400,"No model trained successfully")
    best=max(good,key=lambda r:score_key(task,r)); best_name=best["algorithm"]
    best_preds=all_preds[best_name]; model=all_trained_map[best_name]; fi=[]
    if hasattr(model,"feature_importances_"):
        pairs=sorted(zip(prep.feature_names,model.feature_importances_),key=lambda x:x[1],reverse=True)[:15]
        fi=[{"feature":n,"importance":round(v,5)} for n,v in pairs]
    elif hasattr(model,"coef_"):
        coef=model.coef_
        if coef and isinstance(coef[0],list): vals=[sum(abs(coef[c][j]) for c in range(len(coef)))/max(len(coef),1) for j in range(len(coef[0]))]
        else: vals=[abs(v) for v in coef]
        pairs=sorted(zip(prep.feature_names,vals),key=lambda x:x[1],reverse=True)[:15]
        fi=[{"feature":n,"importance":round(v,5)} for n,v in pairs]
    bm={k:v for k,v in best.items() if k!="algorithm"}
    pred_dist=prediction_distribution(task,best_preds)
    insights=generate_insights(task,req.target,best_name,bm,fi,y_test,best_preds)
    extra: Dict[str,Any] = {}
    if task=="classification":
        extra["confusion_matrix"]=confusion_matrix_data(y_test,best_preds)
        extra["per_class_metrics"]=per_class_metrics(y_test,best_preds)
        cd=Counter(y_train); extra["class_distribution"]=[{"label":str(k),"count":v} for k,v in cd.most_common(12)]
    else:
        extra["residuals"]=residual_samples(y_test,best_preds)
        extra["actual_vs_predicted"]=[{"actual":round(float(a),3),"predicted":round(float(p),3)} for a,p in zip(y_test[:60],best_preds[:60])]
    model_id=uuid.uuid4().hex[:12]
    with open(MODEL_DIR/f"{model_id}.pkl","wb") as f:
        pickle.dump({"preprocessor":prep,"model":model,"target":req.target,"task":task,"feature_columns":feature_cols,"dataset_id":req.dataset_id,"algorithm":best_name},f)
    return {"model_id":model_id,"task":task,"target":req.target,"best_algorithm":best_name,
            "results":results,"feature_importance":fi,"prediction_distribution":pred_dist,
            "insights":insights,"train_size":len(train_rows),"test_size_count":len(test_rows),**extra}

@app.post("/api/ml/predict")
def predict_endpoint(req: PredictRequest):
    mf=MODEL_DIR/f"{req.model_id}.pkl"
    if not mf.exists(): raise HTTPException(404,"Model not found")
    with open(mf,"rb") as f: artifact=pickle.load(f)
    pp=DATA_DIR/f"{req.dataset_id}.csv"
    if not pp.exists(): raise HTTPException(404,"Prediction dataset not found")
    headers,rows=read_csv_file(pp)
    missing=[c for c in artifact["feature_columns"] if c not in headers]
    if missing: raise HTTPException(400,f"Missing columns: {missing}")
    X=artifact["preprocessor"].transform(rows); preds=artifact["model"].predict(X)
    predictions=[{"row":i+1,"prediction":round(p,4) if isinstance(p,float) else str(p)} for i,p in enumerate(preds[:200])]
    pd3=prediction_distribution(artifact["task"],preds)
    return {"model_id":req.model_id,"algorithm":artifact["algorithm"],"target":artifact["target"],
            "task":artifact["task"],"count":len(preds),"predictions":predictions,"prediction_distribution":pd3}
'''

with open(target, 'w', encoding='utf-8') as f:
    f.write(content)

lines = content.count('\n')
print(f'SUCCESS: Written {lines} lines to {target}')
