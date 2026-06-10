import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
)

from mlops_stock_project.config import (
    PROCESSED_DATA_FILE,
    REPORTS_FIGURES_DIR,
)

from mlops_stock_project.features.feature_pipeline import (
    prepare_features,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

from mlops_stock_project.models.simulate_utils import (
    compute_classification_metrics,
    load_model_artifact,
)

logger = get_logger(__name__)


# BASELINE SIMULATION
def simulate_baseline(
    data_path=PROCESSED_DATA_FILE,
):
    logger.info("Running baseline simulation...")

    # LOAD DATA
    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        [
            "Ticker",
            "Date",
        ]
    )

    # LOAD MODEL
    artifact = load_model_artifact()

    model = artifact["model"]

    features = artifact["features"]

    threshold = artifact.get(
        "threshold",
        0.5,
    )

    model_name = artifact.get(
        "model_name",
        "Unknown",
    )

    logger.info(f"Loaded model: {model_name}")

    logger.info(f"Using threshold: {threshold:.4f}")

    # HOLDOUT PERIOD

    test_data = df[df["Date"] >= df["Date"].quantile(0.6)].copy()

    logger.info(f"Test dataset shape: " f"{test_data.shape}")

    # FEATURE MATRIX
    X_test = prepare_features(
        test_data,
        features,
    )

    y_true = test_data["Target"]

    logger.info("Generating probabilities...")

    probabilities = model.predict_proba(X_test)[:, 1]

    y_pred = (probabilities >= threshold).astype(int)

    # METRICS
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    positive_rate = y_true.mean()

    logger.info(
        "Baseline Results | "
        f"F1={metrics['f1']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"BalancedAcc={balanced_acc:.4f} | "
        f"MCC={mcc:.4f}"
    )

    logger.info(f"Positive Class Rate: " f"{positive_rate:.4f}")

    # SECTOR ANALYSIS
    if "Sector" in test_data.columns:

        logger.info("\n========== " "SECTOR PERFORMANCE " "==========")

        for sector in sorted(test_data["Sector"].unique()):

            sector_df = test_data[test_data["Sector"] == sector]

            if len(sector_df) == 0:
                continue

            X_sector = prepare_features(
                sector_df,
                features,
            )

            sector_probs = model.predict_proba(X_sector)[:, 1]

            sector_pred = (sector_probs >= threshold).astype(int)

            sector_f1 = f1_score(
                sector_df["Target"],
                sector_pred,
                zero_division=0,
            )

            logger.info(f"{sector}: " f"F1={sector_f1:.4f} " f"(n={len(sector_df)})")

    # PLOT
    output_file = REPORTS_FIGURES_DIR / "baseline_performance.png"

    metric_names = [
        "F1",
        "Precision",
        "Recall",
        "BalancedAcc",
        "MCC",
    ]

    metric_values = [
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
        balanced_acc,
        mcc,
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        metric_names,
        metric_values,
    )

    for (
        index,
        value,
    ) in enumerate(metric_values):

        plt.text(
            index,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    plt.ylim(
        min(
            -0.1,
            min(metric_values) - 0.05,
        ),
        1.0,
    )

    plt.ylabel("Metric Score")

    plt.title("Baseline Model Performance")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    logger.info(f"Saved baseline " f"performance plot to " f"{output_file}")

    logger.info("Baseline simulation completed.")

    return {
        **metrics,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "positive_rate": positive_rate,
    }


if __name__ == "__main__":
    simulate_baseline()
