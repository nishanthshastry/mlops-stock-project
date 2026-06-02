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


def fetch_stock_data(period="10y"):

    logger.info(
        "Starting multi-stock data fetch..."
    )

    os.makedirs(
        RAW_DATA_DIR,
        exist_ok=True,
    )

    all_data = []

    for ticker in TICKERS:

        try:

            logger.info(
                f"Fetching data for {ticker}"
            )

            data = yf.download(
                ticker,
                period=period,
                auto_adjust=True,
            )

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):

                data.columns = data.columns.get_level_values(0)

            # Reset index
            data = data.reset_index()

            # Keep only required columns
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

            # Add ticker column
            data["Ticker"] = ticker

            # Remove missing rows
            data = data.dropna()

            # Save individual file
            ticker_path = (
                RAW_DATA_DIR / f"{ticker}.csv"
            )

            data.to_csv(
                ticker_path,
                index=False,
            )

            logger.info(
                f"Saved {ticker} "
                f"to {ticker_path}"
            )

            all_data.append(data)

        except Exception as e:

            logger.warning(
                f"Failed to fetch "
                f"{ticker}: {str(e)}"
            )

    # Combine all datasets
    combined_df = pd.concat(
        all_data,
        ignore_index=True,
    )

    combined_df.to_csv(
        COMBINED_RAW_DATA_FILE,
        index=False,
    )

    logger.info(
        f"Combined dataset saved to "
        f"{COMBINED_RAW_DATA_FILE}"
    )

    logger.info(
        f"Combined dataset shape: "
        f"{combined_df.shape}"
    )

    # Track with DVC
    try:

        subprocess.run(
            [
                "dvc",
                "add",
                str(COMBINED_RAW_DATA_FILE),
            ],
            check=True,
        )

        logger.info(
            "Combined raw dataset "
            "tracked with DVC"
        )

    except Exception as e:

        logger.warning(
            f"DVC tracking failed: "
            f"{str(e)}"
        )

    return combined_df


if __name__ == "__main__":

    fetch_stock_data()