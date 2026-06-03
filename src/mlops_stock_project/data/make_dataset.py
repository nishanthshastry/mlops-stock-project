import os
import subprocess

import pandas as pd
import yfinance as yf

from mlops_stock_project.config import (
    RAW_DATA_DIR,
    COMBINED_RAW_DATA_FILE,
    TICKERS,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

logger = get_logger(__name__)

# Market context symbols
MARKET_SYMBOLS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
}


def fetch_market_data(
    period="10y",
):
    """
    Fetch market-wide indicators:
    - SPY
    - QQQ
    - VIX
    """

    logger.info("Fetching market context data...")

    market_frames = []

    for (
        label,
        symbol,
    ) in MARKET_SYMBOLS.items():
        try:
            logger.info(f"Fetching {label}")

            data = yf.download(
                symbol,
                period=period,
                auto_adjust=True,
            )

            # Flatten columns if needed
            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()

            data = data[
                [
                    "Date",
                    "Close",
                    "Volume",
                ]
            ]

            # Rename columns
            data = data.rename(
                columns={
                    "Close": (f"{label}_Close"),
                    "Volume": (f"{label}_Volume"),
                }
            )

            market_frames.append(data)

        except Exception as e:
            logger.warning(f"Failed to fetch {label}: {str(e)}")

    # Merge market datasets
    market_df = market_frames[0]

    for additional_df in market_frames[1:]:
        market_df = market_df.merge(
            additional_df,
            on="Date",
            how="inner",
        )

    logger.info(f"Market dataset shape: {market_df.shape}")

    return market_df


def fetch_stock_data(
    period="10y",
):

    logger.info("Starting multi-stock data fetch...")

    os.makedirs(
        RAW_DATA_DIR,
        exist_ok=True,
    )

    # Fetch market context
    market_df = fetch_market_data(period=period)

    all_data = []

    for ticker in TICKERS:
        try:
            logger.info(f"Fetching data for {ticker}")

            data = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
            )

            # Flatten MultiIndex columns
            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):
                data.columns = data.columns.get_level_values(0)

            # Reset index
            data = data.reset_index()

            # Keep required columns
            data = data[
                [
                    "Date",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]
            ]

            # Add ticker
            data["Ticker"] = ticker

            # Remove missing rows
            data = data.dropna()

            # Merge market context
            data = data.merge(
                market_df,
                on="Date",
                how="left",
            )

            # Remove rows with missing
            # market data
            data = data.dropna()

            # Save individual dataset
            ticker_path = RAW_DATA_DIR / f"{ticker}.csv"

            data.to_csv(
                ticker_path,
                index=False,
            )

            logger.info(f"Saved {ticker} to {ticker_path}")

            all_data.append(data)

        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {str(e)}")

    # Combine all stocks
    combined_df = pd.concat(
        all_data,
        ignore_index=True,
    )

    # Final cleanup
    combined_df = combined_df.sort_values(["Ticker", "Date"]).drop_duplicates()

    combined_df.to_csv(
        COMBINED_RAW_DATA_FILE,
        index=False,
    )

    logger.info(f"Combined dataset saved to {COMBINED_RAW_DATA_FILE}")

    logger.info(f"Combined dataset shape: {combined_df.shape}")

    logger.info(f"Dataset columns:\n{combined_df.columns.tolist()}")

    # DVC tracking
    try:
        subprocess.run(
            [
                "dvc",
                "add",
                str(COMBINED_RAW_DATA_FILE),
            ],
            check=True,
        )

        logger.info("Combined raw dataset tracked with DVC")

    except Exception as e:
        logger.warning(f"DVC tracking failed: {str(e)}")

    return combined_df


if __name__ == "__main__":
    fetch_stock_data()
