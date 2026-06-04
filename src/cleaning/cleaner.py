import pandas as pd
import numpy as np

def load_and_clean(path="data/raw/creditcard.csv"):
    print("Loading dataset...")
    df = pd.read_csv(path)
    
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Fraud cases: {df['Class'].sum():,}")
    print(f"Fraud rate: {df['Class'].mean():.4%}")
    
    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Duplicates removed: {before - len(df)}")
    
    # Check missing values
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Save
    df.to_csv("data/processed/cleaned.csv", index=False)
    print("✓ Saved to data/processed/cleaned.csv")
    return df

if __name__ == "__main__":
    df = load_and_clean()
    print(df.describe())