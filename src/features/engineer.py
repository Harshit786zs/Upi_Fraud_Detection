import pandas as pd
import numpy as np

def engineer_features(path="data/processed/cleaned.csv"):
    print("Engineering features...")
    df = pd.read_csv(path)
    
    # 1. Amount features
    df["amount_log"] = np.log1p(df["Amount"])
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["is_high_amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    
    # 2. Time features
    df["hour"] = (df["Time"] % 86400 // 3600).astype(int)
    df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    
    # 3. Drop original Time and Amount
    df.drop(columns=["Time", "Amount"], inplace=True)
    
    # Save
    df.to_csv("data/processed/features.csv", index=False)
    print(f"✓ Features saved. Shape: {df.shape}")
    print(f"New columns added: amount_log, amount_zscore, is_high_amount, hour, is_night")
    return df

if __name__ == "__main__":
    df = engineer_features()
    print(df.head())