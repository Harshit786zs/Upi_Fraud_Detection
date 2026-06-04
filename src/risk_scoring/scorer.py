import pandas as pd
import numpy as np
import joblib

def get_risk_score(fraud_prob, is_high_amount=0, is_night=0, amount_zscore=0):
    """
    Calculate risk score 0-100 from multiple signals.
    """
    score = 0
    score += fraud_prob * 60          # 60% weight on model probability
    score += is_high_amount * 15      # high amount adds 15 points
    score += is_night * 10            # night transaction adds 10 points
    score += min(abs(amount_zscore) * 5, 15)  # amount deviation max 15 points

    score = min(score, 100)

    if score <= 30:
        level = "LOW"
        action = "ALLOW"
    elif score <= 60:
        level = "MEDIUM"
        action = "CHALLENGE"
    else:
        level = "HIGH"
        action = "BLOCK"

    return {
        "risk_score": round(score, 2),
        "risk_level": level,
        "action": action
    }

def score_dataset(path="data/processed/features.csv"):
    print("Scoring transactions...")
    model = joblib.load("models/saved/XGBoost.pkl")
    scaler = joblib.load("models/saved/scaler.pkl")
    df = pd.read_csv(path)

    X = df.drop(columns=["Class"])
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    results = []
    for i, prob in enumerate(probs):
        r = get_risk_score(
            fraud_prob=prob,
            is_high_amount=df["is_high_amount"].iloc[i],
            is_night=df["is_night"].iloc[i],
            amount_zscore=df["amount_zscore"].iloc[i]
        )
        results.append(r)

    result_df = pd.DataFrame(results)
    result_df["actual"] = df["Class"].values
    result_df.to_csv("data/processed/risk_scores.csv", index=False)

    print("✓ Risk scores saved!")
    print(result_df["action"].value_counts())
    print(result_df["risk_level"].value_counts())

if __name__ == "__main__":
    score_dataset()