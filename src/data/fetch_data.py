import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker="AAPL", period="2y"):
    print(f"Fetching data for {ticker}...")

    data = yf.download(ticker, period=period)

    # Reset index
    data.reset_index(inplace=True)

    # Save to CSV
    os.makedirs("data/raw", exist_ok=True)
    file_path = f"data/raw/{ticker}.csv"
    data.to_csv(file_path, index=False)

    print(f"Data saved to {file_path}")
    return data


if __name__ == "__main__":
    df = fetch_stock_data("AAPL")
    print(df.head())