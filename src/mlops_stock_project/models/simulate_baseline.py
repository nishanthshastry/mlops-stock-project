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
    load_model_artifact,
    create_xgboost_model,
    compute_classification_metrics,
    plot_simulation_metrics,
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

    logger.info(f"Loaded dataset shape: {df.shape}")

    # LOAD ARTIFACT
    artifact = load_model_artifact()

    features = artifact["features"]

    logger.info(f"Using {len(features)} features")

    # SPLIT DATA
    split_date = df["Date"].quantile(0.6)

    train_data = df[df["Date"] < split_date].copy()

    test_data = df[df["Date"] >= split_date].copy()

    # PREPARE FEATURES
    X_train = prepare_features(
        train_data,
        features,
    )

    y_train = train_data["Target"]

    # TRAIN MODEL
    model = create_xgboost_model()

    model.fit(
        X_train,
        y_train,
    )

    # ROLLING EVALUATION
    window_size = 250

    f1_scores = []

    precision_scores = []

    recall_scores = []

    evaluation_steps = []

    for i in range(
        window_size,
        len(test_data),
    ):
        window = test_data.iloc[:i].copy()

        X_window = prepare_features(
            window,
            features,
        )

        y_true = window["Target"]

        y_pred = model.predict(X_window)

        metrics = compute_classification_metrics(
            y_true,
            y_pred,
        )

        f1_scores.append(metrics["f1"])

        precision_scores.append(metrics["precision"])

        recall_scores.append(metrics["recall"])

        evaluation_steps.append(i)

    # FINAL METRICS
    logger.info(f"Final F1: {f1_scores[-1]:.4f}")

    logger.info(f"Final Precision: {precision_scores[-1]:.4f}")

    logger.info(f"Final Recall: {recall_scores[-1]:.4f}")

    # PLOT
    output_file = REPORTS_FIGURES_DIR / "baseline_performance.png"

    plot_simulation_metrics(
        evaluation_steps=evaluation_steps,
        f1_scores=f1_scores,
        precision_scores=precision_scores,
        recall_scores=recall_scores,
        output_file=output_file,
        title="Baseline Model Performance (No Retraining)",
    )

    logger.info("Baseline simulation completed")


if __name__ == "__main__":
    simulate_baseline()
