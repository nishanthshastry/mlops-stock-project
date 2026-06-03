import os
import subprocess

import numpy as np
import pandas as pd

from mlops_stock_project.config import (
    COMBINED_RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

logger = get_logger(__name__)


def build_features(
    input_path=COMBINED_RAW_DATA_FILE,
):

    logger.info("Building market-aware multi-stock features...")

    # Load Dataset
    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    # Clean columns
    df.columns = df.columns.str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Numeric conversion
    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SPY_Close",
        "QQQ_Close",
        "VIX_Close",
        "SPY_Volume",
        "QQQ_Volume",
        "VIX_Volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Sort properly
    df = df.sort_values(["Ticker", "Date"])

    # Group by ticker
    grouped = df.groupby("Ticker")

    # ===================== STOCK FEATURES =====================

    # Returns
    df["Return"] = grouped["Close"].pct_change(fill_method=None)

    # Moving averages
    df["MA_5"] = grouped["Close"].transform(lambda x: x.rolling(5).mean())

    df["MA_10"] = grouped["Close"].transform(lambda x: x.rolling(10).mean())

    # Volatility
    df["Volatility"] = grouped["Close"].transform(lambda x: x.rolling(5).std())

    # Lag features
    df["Lag_1"] = grouped["Return"].shift(1)

    df["Lag_2"] = grouped["Return"].shift(2)

    df["Lag_3"] = grouped["Return"].shift(3)

    # Momentum
    df["Momentum_5"] = grouped["Close"].transform(lambda x: x - x.shift(5))

    # EMA
    df["EMA_10"] = grouped["Close"].transform(lambda x: x.ewm(span=10).mean())

    # RSI
    delta = grouped["Close"].diff()

    gain = (
        delta.where(
            delta > 0,
            0,
        )
        .groupby(df["Ticker"])
        .transform(lambda x: x.rolling(14).mean())
    )

    loss = (
        -delta.where(
            delta < 0,
            0,
        )
        .groupby(df["Ticker"])
        .transform(lambda x: x.rolling(14).mean())
    )

    rs = gain / loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # Volume features
    df["Volume_Change"] = grouped["Volume"].pct_change(fill_method=None)

    df["Volume_MA_5"] = grouped["Volume"].transform(lambda x: x.rolling(5).mean())

    # MACD
    ema_12 = grouped["Close"].transform(lambda x: x.ewm(span=12).mean())

    ema_26 = grouped["Close"].transform(lambda x: x.ewm(span=26).mean())

    df["MACD"] = ema_12 - ema_26

    # Bollinger Bands
    rolling_mean = grouped["Close"].transform(lambda x: x.rolling(20).mean())

    rolling_std = grouped["Close"].transform(lambda x: x.rolling(20).std())

    df["BB_upper"] = rolling_mean + (2 * rolling_std)

    df["BB_lower"] = rolling_mean - (2 * rolling_std)

    # ===================== MARKET CONTEXT FEATURES =====================

    # SPY
    df["SPY_Return"] = df["SPY_Close"].pct_change(fill_method=None)

    df["SPY_MA_5"] = df["SPY_Close"].rolling(5).mean()

    df["SPY_Volatility"] = df["SPY_Close"].rolling(10).std()

    # QQQ
    df["QQQ_Return"] = df["QQQ_Close"].pct_change(fill_method=None)

    df["QQQ_Momentum"] = df["QQQ_Close"] - df["QQQ_Close"].shift(5)

    df["QQQ_MA_10"] = df["QQQ_Close"].rolling(10).mean()

    # VIX
    df["VIX_Return"] = df["VIX_Close"].pct_change(fill_method=None)

    df["VIX_MA_5"] = df["VIX_Close"].rolling(5).mean()

    df["VIX_Level"] = df["VIX_Close"]

    # High volatility regime
    df["High_VIX_Regime"] = (df["VIX_Close"] > 25).astype(int)

    # RELATIVE STRENGTH FEATURES
    df["Relative_SPY_Strength"] = df["Return"] - df["SPY_Return"]

    df["Relative_QQQ_Strength"] = df["Return"] - df["QQQ_Return"]

    # VOLATILITY REGIME FEATURES
    df["Market_Stress"] = np.where(
        (df["VIX_Close"] > 30) & (df["SPY_Return"] < 0),
        1,
        0,
    )

    # TARGET
    future_return = (grouped["Close"].shift(-5) - df["Close"]) / df["Close"]

    # Stronger directional target
    df["Target"] = (future_return > 0.01).astype(int)

    # ===================== CLEANUP =====================

    # Remove infinities
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove missing rows
    df = df.dropna()

    logger.info(f"Final dataset shape: {df.shape}")

    logger.info(f"Target Distribution:\n{df['Target'].value_counts(normalize=True)}")

    # SAVE DATASET
    os.makedirs(
        PROCESSED_DATA_DIR,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA_FILE,
        index=False,
    )

    logger.info(f"Processed features saved to {PROCESSED_DATA_FILE}")

    # DVC TRACKING
    try:
        subprocess.run(
            [
                "dvc",
                "add",
                str(PROCESSED_DATA_FILE),
            ],
            check=True,
        )

        logger.info("Processed dataset tracked with DVC")

    except Exception as e:
        logger.warning(f"DVC tracking failed: {str(e)}")

    return df


if __name__ == "__main__":
    build_features()
