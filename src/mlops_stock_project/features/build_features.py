import os
import pandas as pd
import subprocess

from mlops_stock_project.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
)

from mlops_stock_project.logging_config import get_logger

logger = get_logger(__name__)


def build_features(input_path=RAW_DATA_FILE):
    logger.info("Building features...")

    df = pd.read_csv(input_path)

    # Fix column names
    df.columns = [col.split()[-1] for col in df.columns]

    # Ensure numeric close prices
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Sort by date
    df = df.sort_values("Date")

    # Feature engineering
    df["Return"] = df["Close"].pct_change()

    df["MA_5"] = df["Close"].rolling(window=5).mean()

    df["MA_10"] = df["Close"].rolling(window=10).mean()

    df["Volatility"] = df["Close"].rolling(window=5).std()

    # Binary target
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Remove missing values
    df = df.dropna()

    # Save processed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df.to_csv(PROCESSED_DATA_FILE, index=False)

    try:
        subprocess.run(
            ["dvc", "add", str(PROCESSED_DATA_FILE)],
            check=True,
        )

        logger.info("Processed dataset tracked with DVC")

    except Exception as e:
        logger.warning(f"DVC tracking failed: {str(e)}")

    logger.info(f"Processed features saved to {PROCESSED_DATA_FILE}")

    return df


if __name__ == "__main__":
    build_features()
