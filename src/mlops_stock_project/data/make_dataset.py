import os
import yfinance as yf
import subprocess
from mlops_stock_project.config import RAW_DATA_DIR
from mlops_stock_project.logging_config import get_logger

logger = get_logger(__name__)


def fetch_stock_data(ticker="AAPL", period="2y"):
    logger.info(f"Fetching data for {ticker}")

    data = yf.download(ticker, period=period)

    data.reset_index(inplace=True)

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    output_path = RAW_DATA_DIR / f"{ticker}.csv"

    data.to_csv(output_path, index=False)

    try:
        subprocess.run(
            ["dvc", "add", str(output_path)],
            check=True,
        )

        logger.info("Raw dataset tracked with DVC")

    except Exception as e:
        logger.warning(f"DVC tracking failed: {str(e)}")

    logger.info(f"Data saved to {output_path}")

    return data


if __name__ == "__main__":
    fetch_stock_data()
