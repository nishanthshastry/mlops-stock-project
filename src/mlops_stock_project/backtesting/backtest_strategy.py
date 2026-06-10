import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from mlops_stock_project.config import (
    MODEL_FILE,
    PROCESSED_DATA_FILE,
    PROJECT_ROOT,
)

from mlops_stock_project.features.feature_pipeline import (
    prepare_features,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

logger = get_logger(__name__)

# REPORT DIRECTORIES
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# BACKTEST CONFIG
INITIAL_CAPITAL = 10000

TRANSACTION_COST = 0.001

TRADING_DAYS = 252


# MAIN BACKTEST
def backtest_strategy():

    logger.info("Starting trading strategy backtest...")

    # LOAD MODEL
    artifact = joblib.load(MODEL_FILE)

    model = artifact["model"]

    threshold = artifact["threshold"]

    features = artifact["features"]

    model_name = artifact.get(
        "model_name",
        "Unknown",
    )

    logger.info(f"Loaded model: {model_name}")

    logger.info(f"Decision threshold: {threshold:.4f}")

    # LOAD DATA
    df = pd.read_csv(PROCESSED_DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    # SORT BY TICKER + DATE
    df = df.sort_values(["Ticker", "Date"])

    logger.info(f"Loaded dataset shape: {df.shape}")

    # DATE-BASED SPLIT
    split_date = df["Date"].quantile(0.8)

    test_df = df[df["Date"] >= split_date].copy()

    logger.info(f"Test dataset shape: {test_df.shape}")

    # FEATURE PREPARATION
    X_test = prepare_features(
        test_df,
        features,
    )

    # HANDLE PIPELINES
    actual_model = model

    if isinstance(
        model,
        Pipeline,
    ):
        scaler = model.named_steps["scaler"]

        X_test_transformed = scaler.transform(X_test)

        actual_model = model.named_steps["model"]

    else:
        X_test_transformed = X_test

    # MODEL PREDICTIONS
    probabilities = actual_model.predict_proba(X_test_transformed)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    test_df["Prediction"] = predictions

    test_df["Probability"] = probabilities

    # NEXT-DAY RETURNS
    test_df["Future_Return"] = (
        test_df.groupby("Ticker")["Close"].shift(-1) / test_df["Close"] - 1
    )

    # Remove invalid rows
    test_df = test_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    test_df = test_df.dropna(subset=["Future_Return"])

    # LONG-ONLY STRATEGY
    test_df["Strategy_Return"] = np.where(
        test_df["Prediction"] == 1,
        test_df["Future_Return"],
        0,
    )

    # TRANSACTION COSTS
    test_df["Trade_Change"] = test_df.groupby("Ticker")["Prediction"].diff().abs()

    test_df["Trade_Change"] = test_df["Trade_Change"].fillna(0)

    test_df["Transaction_Cost"] = test_df["Trade_Change"] * TRANSACTION_COST

    test_df["Net_Strategy_Return"] = (
        test_df["Strategy_Return"] - test_df["Transaction_Cost"]
    )

    # DAILY PORTFOLIO RETURNS
    daily_strategy_returns = test_df.groupby("Date")["Net_Strategy_Return"].mean()

    daily_market_returns = test_df.groupby("Date")["Future_Return"].mean()

    # EQUITY CURVES
    strategy_equity = INITIAL_CAPITAL * (1 + daily_strategy_returns).cumprod()

    market_equity = INITIAL_CAPITAL * (1 + daily_market_returns).cumprod()

    # PERFORMANCE METRICS
    strategy_total_return = strategy_equity.iloc[-1] / INITIAL_CAPITAL - 1

    market_total_return = market_equity.iloc[-1] / INITIAL_CAPITAL - 1

    # Sharpe Ratio
    if daily_strategy_returns.std() != 0:
        sharpe_ratio = (
            daily_strategy_returns.mean() / daily_strategy_returns.std()
        ) * np.sqrt(TRADING_DAYS)

    else:
        sharpe_ratio = 0

    # Max Drawdown
    rolling_max = strategy_equity.cummax()

    drawdown = (strategy_equity - rolling_max) / rolling_max

    max_drawdown = drawdown.min()

    # Win Rate
    profitable_days = daily_strategy_returns[daily_strategy_returns > 0]

    if len(daily_strategy_returns) > 0:
        win_rate = len(profitable_days) / len(daily_strategy_returns)

    else:
        win_rate = 0

    # LOG RESULTS
    logger.info("\n========== BACKTEST RESULTS ==========")

    logger.info(f"Strategy Return: {strategy_total_return:.2%}")

    logger.info(f"Market Return: {market_total_return:.2%}")

    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    logger.info(f"Max Drawdown: {max_drawdown:.2%}")

    logger.info(f"Win Rate: {win_rate:.2%}")

    if "Sector" in test_df.columns:

        logger.info("\n========== SECTOR RETURNS ==========")

        sector_results = []

        for sector in sorted(test_df["Sector"].unique()):

            sector_df = test_df[test_df["Sector"] == sector]

            sector_daily_returns = sector_df.groupby("Date")[
                "Net_Strategy_Return"
            ].mean()

            sector_return = (1 + sector_daily_returns).cumprod().iloc[-1] - 1

            sector_results.append(
                {
                    "Sector": sector,
                    "Return": round(
                        float(sector_return),
                        4,
                    ),
                    "Samples": len(sector_df),
                }
            )

        sector_results_df = pd.DataFrame(sector_results)

        logger.info(f"\n{sector_results_df}")

        sector_results_df.to_csv(
            FIGURES_DIR / "sector_backtest_returns.csv",
            index=False,
        )

    # EQUITY CURVE PLOT
    plt.figure(figsize=(14, 7))

    plt.plot(
        strategy_equity.index,
        strategy_equity.values,
        label="Strategy",
    )

    plt.plot(
        market_equity.index,
        market_equity.values,
        label="Buy & Hold",
    )

    plt.title("Strategy vs Buy-and-Hold")

    plt.xlabel("Date")

    plt.ylabel("Portfolio Value")

    plt.legend()

    plt.grid(True)

    equity_path = FIGURES_DIR / "strategy_vs_market.png"

    plt.savefig(
        equity_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    logger.info(f"Saved equity curve to {equity_path}")

    logger.info("Backtesting completed.")

    return {
        "strategy_return": strategy_total_return,
        "market_return": market_total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


if __name__ == "__main__":
    backtest_strategy()
