from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os
import time
from src.risk_scoring.scorer import get_risk_score

app = FastAPI(title="UPI Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ACTIVE_MODEL_NAME = "XGBoost"

model = None
scaler = None
model_loaded = False
scaler_loaded = False

try:
    model = joblib.load(f"models/saved/{ACTIVE_MODEL_NAME}.pkl")
    model_loaded = True
except Exception as e:
    print(f"⚠ Could not load model: {e}")

try:
    scaler = joblib.load("models/saved/scaler.pkl")
    scaler_loaded = True
except Exception as e:
    print(f"⚠ Could not load scaler: {e}")

# Fallback metrics used only if models/saved/metrics.json hasn't been
# generated yet (run `python -m src.models.train` to produce real numbers
# from your test set). ROC-AUC values below match the originally reported
# figures; precision/recall/F1 are placeholders until retrained.
FALLBACK_METRICS = {
    "LogisticRegression": {"accuracy": 0.9634, "precision": 0.89, "recall": 0.81, "f1_score": 0.85, "roc_auc": 0.9634, "inference_time_ms": 0.112},
    "RandomForest":       {"accuracy": 0.9607, "precision": 0.93, "recall": 0.85, "f1_score": 0.89, "roc_auc": 0.9607, "inference_time_ms": 7.751},
    "XGBoost":            {"accuracy": 0.9794, "precision": 0.95, "recall": 0.90, "f1_score": 0.92, "roc_auc": 0.9794, "inference_time_ms": 0.459},
}


def _load_metrics():
    path = "models/saved/metrics.json"
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_source"] = "measured"
            return data
        except Exception:
            pass
    metrics = {k: dict(v) for k, v in FALLBACK_METRICS.items()}
    metrics["_source"] = "fallback"
    return metrics


def _measure_inference_ms(m, n_runs=100):
    """Real single-transaction latency for whichever model is active."""
    if m is None or scaler is None:
        return None
    try:
        sample = scaler.transform(np.zeros((1, 33)))
        for _ in range(5):
            m.predict_proba(sample)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            m.predict_proba(sample)
        return round((time.perf_counter() - t0) / n_runs * 1000, 3)
    except Exception:
        return None


_active_inference_ms = _measure_inference_ms(model)

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    amount_log: float
    amount_zscore: float
    is_high_amount: int
    hour: int
    is_night: int

@app.get("/")
def root():
    return {"message": "UPI Fraud Detection API running"}

@app.get("/health")
def health():
    return {
        "status": "ok" if model_loaded and scaler_loaded else "degraded",
        "backend": True,
        "model_loaded": model_loaded,
        "scaler_loaded": scaler_loaded,
        "active_model": ACTIVE_MODEL_NAME,
    }

@app.get("/stats")
def stats():
    metrics = _load_metrics()
    source = metrics.pop("_source", "fallback")
    # Prefer a freshly measured latency for the currently-loaded model.
    if ACTIVE_MODEL_NAME in metrics and _active_inference_ms is not None:
        metrics[ACTIVE_MODEL_NAME] = {
            **metrics[ACTIVE_MODEL_NAME],
            "inference_time_ms": _active_inference_ms,
        }
    return {
        "active_model": ACTIVE_MODEL_NAME,
        "metrics_source": source,  # "measured" (from train.py) or "fallback"
        "models": metrics,
    }

@app.post("/predict")
def predict(txn: Transaction):
    features = np.array([[
        txn.V1, txn.V2, txn.V3, txn.V4, txn.V5,
        txn.V6, txn.V7, txn.V8, txn.V9, txn.V10,
        txn.V11, txn.V12, txn.V13, txn.V14, txn.V15,
        txn.V16, txn.V17, txn.V18, txn.V19, txn.V20,
        txn.V21, txn.V22, txn.V23, txn.V24, txn.V25,
        txn.V26, txn.V27, txn.V28, txn.amount_log,
        txn.amount_zscore, txn.is_high_amount,
        txn.hour, txn.is_night
    ]])
    scaled = scaler.transform(features)
    prob = float(model.predict_proba(scaled)[0][1])
    risk = get_risk_score(prob, txn.is_high_amount,
                          txn.is_night, txn.amount_zscore)
    return {"fraud_probability": round(prob, 4), **risk}