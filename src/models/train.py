import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib, os

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

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test_sc)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_sc)[:,1])
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f}")
        joblib.dump(model, f"models/saved/{name}.pkl")
        results[name] = auc

    joblib.dump(scaler, "models/saved/scaler.pkl")
    print("\n✓ All models saved!")
    print("\nROC-AUC Summary:")
    for name, auc in results.items():
        print(f"  {name}: {auc:.4f}")

if __name__ == "__main__":
    train_models()