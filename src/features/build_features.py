import pandas as pd
import os

def build_features(input_path="data/raw/AAPL.csv"):
    print("Building features...")

    df = pd.read_csv(input_path)

    # Fix column names (flatten multi-level columns)
    df.columns = [col.split()[-1] for col in df.columns]

    # Ensure numeric columns
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Sort by date
    df = df.sort_values("Date")

    # Create features
    df["Return"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["Volatility"] = df["Close"].rolling(window=5).std()

    # Create target
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop missing values
    df = df.dropna()

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/AAPL_features.csv"
    df.to_csv(output_path, index=False)

    print(f"Features saved to {output_path}")
    return df


if __name__ == "__main__":
    df = build_features()
    print(df.head())