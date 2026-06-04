from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from src.risk_scoring.scorer import get_risk_score

app = FastAPI(title="UPI Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/saved/XGBoost.pkl")
scaler = joblib.load("models/saved/scaler.pkl")

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
    return {"status": "ok"}

@app.get("/stats")
def stats():
    return {
        "models": {
            "LogisticRegression": 0.9634,
            "RandomForest": 0.9607,
            "XGBoost": 0.9794
        }
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