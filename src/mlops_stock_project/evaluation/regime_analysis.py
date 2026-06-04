import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

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

TRADING_DAYS = 252


# REGIME CLASSIFICATION


def classify_market_regime(
    vix_level,
):

    if vix_level < 20:
        return "Low Volatility"

    elif vix_level < 30:
        return "Medium Volatility"

    return "High Volatility"


# MAIN REGIME ANALYSIS


def run_regime_analysis():

    logger.info("Starting regime-based evaluation...")

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

    # LOAD DATA

    df = pd.read_csv(PROCESSED_DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Ticker", "Date"])

    logger.info(f"Loaded dataset shape: {df.shape}")

    # DATE-BASED TEST SPLIT

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

    # FUTURE RETURNS

    test_df["Future_Return"] = (
        test_df.groupby("Ticker")["Close"].shift(-1) / test_df["Close"] - 1
    )

    test_df = test_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    test_df = test_df.dropna(subset=["Future_Return"])

    # STRATEGY RETURNS

    test_df["Strategy_Return"] = np.where(
        test_df["Prediction"] == 1,
        test_df["Future_Return"],
        0,
    )

    # REGIME LABELING

    test_df["Market_Regime"] = test_df["VIX_Level"].apply(classify_market_regime)

    logger.info(f"\nRegime Distribution:\n{test_df['Market_Regime'].value_counts()}")

    # REGIME PERFORMANCE

    regime_results = []

    for regime in sorted(test_df["Market_Regime"].unique()):
        regime_df = test_df[test_df["Market_Regime"] == regime]

        if len(regime_df) == 0:
            continue

        y_true = regime_df["Target"]

        y_pred = regime_df["Prediction"]

        returns = regime_df["Strategy_Return"]

        # Metrics
        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        avg_return = returns.mean()

        # Sharpe
        if returns.std() != 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS)

        else:
            sharpe_ratio = 0

        # Win Rate
        profitable_trades = returns[returns > 0]

        if len(returns) > 0:
            win_rate = len(profitable_trades) / len(returns)

        else:
            win_rate = 0

        # Save results
        regime_results.append(
            {
                "Regime": regime,
                "Samples": len(regime_df),
                "Accuracy": round(
                    accuracy,
                    4,
                ),
                "Precision": round(
                    precision,
                    4,
                ),
                "Recall": round(
                    recall,
                    4,
                ),
                "F1_Score": round(
                    f1,
                    4,
                ),
                "Average_Return": round(
                    avg_return,
                    6,
                ),
                "Sharpe_Ratio": round(
                    sharpe_ratio,
                    4,
                ),
                "Win_Rate": round(
                    win_rate,
                    4,
                ),
            }
        )

    # RESULTS DATAFRAME

    regime_results_df = pd.DataFrame(regime_results)

    logger.info("\n========== REGIME PERFORMANCE ==========\n")

    logger.info(f"\n{regime_results_df}")

    # SAVE CSV

    output_csv = FIGURES_DIR / "regime_performance.csv"

    regime_results_df.to_csv(
        output_csv,
        index=False,
    )

    logger.info(f"Saved regime performance CSV to {output_csv}")

    # F1 SCORE PLOT

    plt.figure(figsize=(10, 6))

    plt.bar(
        regime_results_df["Regime"],
        regime_results_df["F1_Score"],
    )

    plt.title("F1 Score by Market Regime")

    plt.ylabel("F1 Score")

    plt.grid(True)

    f1_plot_path = FIGURES_DIR / "regime_f1_scores.png"

    plt.savefig(
        f1_plot_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    logger.info(f"Saved regime F1 plot to {f1_plot_path}")

    logger.info("Regime analysis completed.")

    return regime_results_df


if __name__ == "__main__":
    run_regime_analysis()
