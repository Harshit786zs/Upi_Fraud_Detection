import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import os

def explain_model(path="data/processed/features.csv"):
    print("Loading model and data...")
    model = joblib.load("models/saved/XGBoost.pkl")
    scaler = joblib.load("models/saved/scaler.pkl")
    
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    
    # Scale and take sample
    X_scaled = scaler.transform(X)
    X_sample = X_scaled[:500]
    
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    os.makedirs("reports", exist_ok=True)
    
    # Summary plot
    print("Generating summary plot...")
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=X.columns.tolist(),
        show=False
    )
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=150)
    plt.close()
    
    print("✓ SHAP summary plot saved to reports/shap_summary.png")
    
    # Top features
    mean_shap = pd.DataFrame({
        "feature": X.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 most important features:")
    print(mean_shap.head(10).to_string(index=False))

if __name__ == "__main__":
    explain_model()