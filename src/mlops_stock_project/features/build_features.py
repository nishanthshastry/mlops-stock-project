import os
import subprocess

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

    logger.info(
        "Building multi-stock features..."
    )

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
    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    # Ensure numeric
    numeric_cols = [
        "Close",
        "Volume",
        "Open",
        "High",
        "Low",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Sort properly
    df = df.sort_values(
        ["Ticker", "Date"]
    )

    # Feature Engineering Per Ticker
    grouped = df.groupby("Ticker")

    # Returns
    df["Return"] = grouped["Close"].pct_change(
        fill_method=None
    )

    # Moving averages
    df["MA_5"] = (
        grouped["Close"]
        .transform(
            lambda x: x.rolling(5).mean()
        )
    )

    df["MA_10"] = (
        grouped["Close"]
        .transform(
            lambda x: x.rolling(10).mean()
        )
    )

    # Volatility
    df["Volatility"] = (
        grouped["Close"]
        .transform(
            lambda x: x.rolling(5).std()
        )
    )

    # Lag features
    df["Lag_1"] = grouped["Return"].shift(1)

    df["Lag_2"] = grouped["Return"].shift(2)

    df["Lag_3"] = grouped["Return"].shift(3)

    # Momentum
    df["Momentum_5"] = (
        grouped["Close"]
        .transform(
            lambda x: x - x.shift(5)
        )
    )

    # EMA
    df["EMA_10"] = (
        grouped["Close"]
        .transform(
            lambda x: (
                x.ewm(span=10).mean()
            )
        )
    )

    # RSI
    delta = grouped["Close"].diff()

    gain = (
        delta.where(delta > 0, 0)
        .groupby(df["Ticker"])
        .transform(
            lambda x: x.rolling(14).mean()
        )
    )

    loss = (
        -delta.where(delta < 0, 0)
        .groupby(df["Ticker"])
        .transform(
            lambda x: x.rolling(14).mean()
        )
    )

    rs = gain / loss

    df["RSI"] = (
        100 - (100 / (1 + rs))
    )

    # Volume features
    df["Volume_Change"] = (
        grouped["Volume"]
        .pct_change(fill_method=None)
    )

    df["Volume_MA_5"] = (
        grouped["Volume"]
        .transform(
            lambda x: x.rolling(5).mean()
        )
    )

    # MACD
    ema_12 = (
        grouped["Close"]
        .transform(
            lambda x: x.ewm(span=12).mean()
        )
    )

    ema_26 = (
        grouped["Close"]
        .transform(
            lambda x: x.ewm(span=26).mean()
        )
    )

    df["MACD"] = ema_12 - ema_26

    # Bollinger Bands
    rolling_mean = (
        grouped["Close"]
        .transform(
            lambda x: x.rolling(20).mean()
        )
    )

    rolling_std = (
        grouped["Close"]
        .transform(
            lambda x: x.rolling(20).std()
        )
    )

    df["BB_upper"] = (
        rolling_mean
        + (2 * rolling_std)
    )

    df["BB_lower"] = (
        rolling_mean
        - (2 * rolling_std)
    )

    # Target Variable
    # Future 5-day return
    future_return = (
        (
            grouped["Close"].shift(-5)
            - df["Close"]
        )
        / df["Close"]
    )

    # Directional prediction target
    df["Target"] = (
        future_return > 0
    ).astype(int)

    logger.info(
        f"Target Distribution:\n"
        f"{df['Target'].value_counts(normalize=True)}"
    )

    # Remove Missing Rows
    df = df.dropna()

    logger.info(
        f"Final dataset shape: "
        f"{df.shape}"
    )

    # Save Processed Dataset
    os.makedirs(
        PROCESSED_DATA_DIR,
        exist_ok=True,
    )

    df.to_csv(
        PROCESSED_DATA_FILE,
        index=False,
    )

    logger.info(
        f"Processed features saved to "
        f"{PROCESSED_DATA_FILE}"
    )

    # DVC Tracking
    try:

        subprocess.run(
            [
                "dvc",
                "add",
                str(PROCESSED_DATA_FILE),
            ],
            check=True,
        )

        logger.info(
            "Processed dataset tracked "
            "with DVC"
        )

    except Exception as e:

        logger.warning(
            f"DVC tracking failed: "
            f"{str(e)}"
        )

    return df


if __name__ == "__main__":

    build_features()