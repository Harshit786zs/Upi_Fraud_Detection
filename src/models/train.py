import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib, os, json, time

def train_models(path="data/processed/features.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"Fraud cases: {y.sum()} / {len(y)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_sc, y_train)
    print(f"After SMOTE: {y_res.value_counts().to_dict()}")

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    os.makedirs("models/saved", exist_ok=True)
    results = {}
    metrics = {}

    # Single-row sample used to time real single-transaction inference latency,
    # matching how the API scores one transaction at a time.
    sample = X_test_sc[:1]

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Warm up, then time N single-row predictions to get a stable avg (ms).
        for _ in range(5):
            model.predict_proba(sample)
        n_runs = 200
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model.predict_proba(sample)
        inference_ms = (time.perf_counter() - t0) / n_runs * 1000

        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f} | Accuracy: {acc:.4f} | "
              f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | "
              f"Inference: {inference_ms:.3f} ms")

        joblib.dump(model, f"models/saved/{name}.pkl")
        results[name] = auc
        metrics[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "inference_time_ms": round(inference_ms, 3),
        }

    joblib.dump(scaler, "models/saved/scaler.pkl")

    with open("models/saved/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✓ All models saved!")
    print("✓ Full metrics written to models/saved/metrics.json")
    print("\nROC-AUC Summary:")
    for name, auc in results.items():
        print(f"  {name}: {auc:.4f}")

if __name__ == "__main__":
    train_models()