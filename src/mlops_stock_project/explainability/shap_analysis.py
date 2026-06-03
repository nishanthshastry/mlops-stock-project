import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from mlops_stock_project.config import (
    MODEL_FILE,
    PROCESSED_DATA_FILE,
    PROJECT_ROOT,
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
    sample_size=2000,
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

    # RECREATE TICKER DUMMIES

    if "Ticker" in df.columns:
        logger.info("Recreating ticker dummy variables...")

        ticker_dummies = pd.get_dummies(
            df["Ticker"],
            prefix="Ticker",
        )

        df = pd.concat(
            [df, ticker_dummies],
            axis=1,
        )

    # ENSURE FEATURE CONSISTENCY

    missing_features = [feature for feature in features if feature not in df.columns]

    if missing_features:
        logger.warning(f"Missing features detected: {missing_features}")

        logger.warning("Adding missing features with zero values...")

        for feature in missing_features:
            df[feature] = 0

    # FINAL FEATURE MATRIX

    X = df[features].copy()

    # Convert booleans to integers
    bool_columns = X.select_dtypes(include=["bool"]).columns

    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype(int)

    # Force numeric conversion
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Replace infinities
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Fill missing values
    X = X.fillna(0)

    # Convert everything to float32
    X = X.astype("float32")

    logger.info(f"Final SHAP feature matrix: {X.shape}")

    logger.info(f"Feature dtype summary:\n{X.dtypes.value_counts()}")

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

    # Final numeric enforcement
    X_transformed = np.array(
        X_transformed,
        dtype=np.float32,
    )

    # CREATE SHAP EXPLAINER

    logger.info("Building SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(actual_model)

    shap_values = explainer.shap_values(X_transformed)

    logger.info("SHAP values generated")

    # SHAP SUMMARY PLOT

    logger.info("Generating SHAP summary plot...")

    plt.figure(figsize=(14, 10))

    shap.summary_plot(
        shap_values,
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

    # SHAP BAR PLOT

    logger.info("Generating SHAP bar plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
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
            "importance": (np.abs(shap_values).mean(axis=0)),
        }
    )

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    )

    logger.info(f"\nTop SHAP Features:\n{importance_df.head(15)}")

    # SAVE FEATURE IMPORTANCE

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
