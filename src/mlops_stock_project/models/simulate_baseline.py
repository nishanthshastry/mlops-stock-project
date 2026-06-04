import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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

    df = df.sort_values(["Ticker", "Date"])

    # LOAD SAVED MODEL
    artifact = load_model_artifact()

    model = artifact["model"]

    features = artifact["features"]

    model_name = artifact.get(
        "model_name",
        "Unknown",
    )

    logger.info(f"Loaded model: {model_name}")

    # HOLDOUT TEST PERIOD
    test_data = df[df["Date"] >= df["Date"].quantile(0.6)].copy()

    logger.info(f"Test dataset shape: {test_data.shape}")

    # FEATURE MATRIX
    X_test = prepare_features(
        test_data,
        features,
    )

    y_true = test_data["Target"]

    logger.info("Generating predictions...")

    y_pred = model.predict(X_test)

    # METRICS
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
    )

    logger.info(
        "Baseline Results | "
        f"F1={metrics['f1']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f}"
    )

    # VISUALIZATION
    output_file = REPORTS_FIGURES_DIR / "baseline_performance.png"

    metric_names = [
        "F1",
        "Precision",
        "Recall",
    ]

    metric_values = [
        metrics["f1"],
        metrics["precision"],
        metrics["recall"],
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        metric_names,
        metric_values,
    )

    for index, value in enumerate(metric_values):
        plt.text(
            index,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    plt.ylim(0, 1)

    plt.ylabel("Score")

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

    logger.info(f"Saved baseline performance plot to {output_file}")

    logger.info("Baseline simulation completed.")

    return metrics


if __name__ == "__main__":
    simulate_baseline()
