import joblib
import shap
import pandas as pd
import numpy as np
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

# REPORT PATHS

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# MAIN SHAP ANALYSIS


def run_shap_analysis(
    sample_size=1000,
):
    logger.info("Starting SHAP analysis...")

    # LOAD MODEL ARTIFACT

    artifact = joblib.load(MODEL_FILE)

    model = artifact["model"]

    features = artifact["features"]

    model_name = artifact.get(
        "model_name",
        "Unknown",
    )

    logger.info(f"Loaded model: {model_name}")

    logger.info(f"Model feature count: {len(features)}")

    # LOAD DATASET

    df = pd.read_csv(PROCESSED_DATA_FILE)

    logger.info(f"Loaded dataset shape: {df.shape}")

    # USE CENTRALIZED FEATURE PIPELINE

    X = prepare_features(
        df,
        features,
    )

    logger.info(f"Final SHAP feature matrix: {X.shape}")

    # SAMPLE DATA

    if len(X) > sample_size:
        X_sample = X.sample(
            sample_size,
            random_state=42,
        )
    else:
        X_sample = X.copy()

    logger.info(f"SHAP sample size: {len(X_sample)}")

    # HANDLE PIPELINES

    actual_model = model

    if isinstance(
        model,
        Pipeline,
    ):
        logger.info("Detected sklearn Pipeline model")

        actual_model = model.named_steps["model"]

        scaler = model.named_steps["scaler"]

        X_transformed = scaler.transform(X_sample)

    else:
        X_transformed = X_sample

    X_transformed = np.asarray(
        X_transformed,
        dtype=np.float32,
    )

    # BUILD EXPLAINER

    logger.info("Building SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(actual_model)

    shap_values = explainer(X_transformed)

    if hasattr(shap_values, "values"):
        shap_values_array = shap_values.values
    else:
        shap_values_array = shap_values

    logger.info("SHAP values generated")

    # SUMMARY PLOT

    logger.info("Generating SHAP summary plot...")

    plt.figure(figsize=(14, 10))

    shap.summary_plot(
        shap_values_array,
        X_sample,
        show=False,
    )

    summary_path = FIGURES_DIR / "shap_summary.png"

    plt.tight_layout()

    plt.savefig(
        summary_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    logger.info(f"Saved SHAP summary plot to {summary_path}")

    # BAR PLOT

    logger.info("Generating SHAP bar plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values_array,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20,
    )

    bar_path = FIGURES_DIR / "shap_bar.png"

    plt.tight_layout()

    plt.savefig(
        bar_path,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    logger.info(f"Saved SHAP bar plot to {bar_path}")

    # FEATURE IMPORTANCE

    importance_df = pd.DataFrame(
        {
            "feature": features,
            "importance": np.abs(shap_values_array).mean(axis=0),
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    market_features = [
        "SPY_Return",
        "SPY_MA_5",
        "SPY_Volatility",
        "QQQ_Return",
        "QQQ_Momentum",
        "QQQ_MA_10",
        "VIX_Return",
        "VIX_MA_5",
        "VIX_Level",
    ]

    sector_features = [c for c in importance_df["feature"] if "Sector" in c]

    market_score = importance_df[importance_df["feature"].isin(market_features)][
        "importance"
    ].sum()

    sector_score = importance_df[importance_df["feature"].isin(sector_features)][
        "importance"
    ].sum()

    logger.info(f"Market Feature Importance: {market_score:.4f}")

    logger.info(f"Sector Feature Importance: {sector_score:.4f}")

    sector_importance = importance_df[importance_df["feature"].str.contains("Sector")]

    logger.info(f"\nSector SHAP Importance:\n" f"{sector_importance}")

    sector_csv = FIGURES_DIR / "sector_shap_importance.csv"

    sector_importance.to_csv(
        sector_csv,
        index=False,
    )

    logger.info(f"\nTop SHAP Features:\n{importance_df.head(15)}")

    importance_csv = FIGURES_DIR / "shap_feature_importance.csv"

    importance_df.to_csv(
        importance_csv,
        index=False,
    )

    logger.info(f"Saved SHAP feature importance CSV to {importance_csv}")

    logger.info("SHAP analysis completed.")

    return importance_df


if __name__ == "__main__":
    run_shap_analysis()
