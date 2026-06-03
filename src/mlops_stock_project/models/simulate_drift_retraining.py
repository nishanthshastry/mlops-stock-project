import pandas as pd

from mlops_stock_project.config import (
    PROCESSED_DATA_FILE,
    REPORTS_FIGURES_DIR,
)

from mlops_stock_project.features.feature_pipeline import (
    prepare_features,
)

from mlops_stock_project.monitoring.drift import (
    detect_drift,
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


# DRIFT RETRAINING


def simulate_drift_retraining(
    data_path=PROCESSED_DATA_FILE,
):

    logger.info("Running drift-triggered retraining simulation...")

    # LOAD DATA
    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Ticker", "Date"])

    artifact = load_model_artifact()

    features = artifact["features"]

    split_date = df["Date"].quantile(0.6)

    train_data = df[df["Date"] < split_date].copy()

    test_data = df[df["Date"] >= split_date].copy()

    X_train = prepare_features(
        train_data,
        features,
    )

    y_train = train_data["Target"]

    model = create_xgboost_model()

    model.fit(
        X_train,
        y_train,
    )

    # CONFIG
    window_size = 250

    drift_check_interval = 250

    f1_scores = []

    precision_scores = []

    recall_scores = []

    evaluation_steps = []

    drift_events = []

    # ROLLING LOOP
    for i in range(
        window_size,
        len(test_data),
    ):
        # DRIFT CHECK
        if i % drift_check_interval == 0:
            historical_window = test_data.iloc[
                max(0, i - drift_check_interval) : i
            ].copy()

            current_window = test_data.iloc[
                i : min(i + drift_check_interval, len(test_data))
            ].copy()

            if len(historical_window) > 50 and len(current_window) > 50:
                drift_detected = detect_drift(
                    historical_window,
                    current_window,
                )

                if drift_detected:
                    logger.info(f"Drift detected at step {i}")

                    drift_events.append(i)

                    retrain_data = pd.concat([train_data, test_data.iloc[:i]]).copy()

                    X_retrain = prepare_features(
                        retrain_data,
                        features,
                    )

                    y_retrain = retrain_data["Target"]

                    model.fit(
                        X_retrain,
                        y_retrain,
                    )

        # EVALUATE
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

    logger.info(f"Final F1: {f1_scores[-1]:.4f}")

    logger.info(f"Total Drift Events: {len(drift_events)}")

    # PLOT
    output_file = REPORTS_FIGURES_DIR / "drift_retraining_performance.png"

    plot_simulation_metrics(
        evaluation_steps=evaluation_steps,
        f1_scores=f1_scores,
        precision_scores=precision_scores,
        recall_scores=recall_scores,
        output_file=output_file,
        title="Drift-Triggered Retraining Performance",
        vertical_markers=drift_events,
    )

    logger.info("Drift retraining simulation completed")


if __name__ == "__main__":
    simulate_drift_retraining()
