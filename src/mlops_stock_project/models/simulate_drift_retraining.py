import numpy as np
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


# DRIFT-TRIGGERED RETRAINING SIMULATION
def simulate_drift_retraining(
    data_path=PROCESSED_DATA_FILE,
):
    logger.info("Running drift-triggered retraining simulation...")

    # LOAD DATA
    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        [
            "Ticker",
            "Date",
        ]
    )

    logger.info(f"Dataset rows: {len(df)}")

    # LOAD TRAINED MODEL ARTIFACT
    artifact = load_model_artifact()

    features = artifact["features"]

    logger.info(f"Using {len(features)} features")

    # TIME SPLIT
    split_date = df["Date"].quantile(0.60)

    train_data = df[df["Date"] < split_date].copy()

    test_data = df[df["Date"] >= split_date].copy()

    logger.info(f"Train rows: {len(train_data)}")

    logger.info(f"Test rows: {len(test_data)}")

    # INITIAL TRAINING
    X_train = prepare_features(
        train_data,
        features,
    )

    y_train = train_data["Target"]

    model = create_xgboost_model(
        fast_mode=True,
    )

    model.fit(
        X_train,
        y_train,
    )

    # SIMULATION CONFIG
    window_size = 500

    drift_check_interval = 1000

    evaluation_window = 1000

    f1_scores = []

    precision_scores = []

    recall_scores = []

    evaluation_steps = []

    drift_events = []

    drift_scores = []

    logger.info(f"Evaluating {len(test_data)} " f"rows in chunks of {window_size}")

    # SIMULATION LOOP
    for i in range(
        window_size,
        len(test_data),
        window_size,
    ):

        current_drift_score = np.nan

        # DRIFT DETECTION
        if i % drift_check_interval == 0:

            historical_window = test_data.iloc[
                max(
                    0,
                    i - drift_check_interval,
                ) : i
            ]

            current_window = test_data.iloc[
                i : min(
                    i + drift_check_interval,
                    len(test_data),
                )
            ]

            if len(historical_window) > 50 and len(current_window) > 50:

                drift_result = detect_drift(
                    historical_window,
                    current_window,
                )

                drift_detected = drift_result["overall_drift_detected"]

                current_drift_score = drift_result.get(
                    "overall_drift_score",
                    0,
                )

                logger.info(f"Step={i} | " f"Drift Score=" f"{current_drift_score:.4f}")

                # RETRAIN
                if drift_detected:

                    logger.info(f"Drift detected at " f"step {i}")

                    drift_events.append(i)

                    retrain_data = pd.concat(
                        [
                            train_data,
                            test_data.iloc[:i],
                        ]
                    )

                    X_retrain = prepare_features(
                        retrain_data,
                        features,
                    )

                    y_retrain = retrain_data["Target"]

                    model.fit(
                        X_retrain,
                        y_retrain,
                    )

                    logger.info(f"Retrained on " f"{len(retrain_data)} rows")

        drift_scores.append(current_drift_score)

        # ROLLING EVALUATION
        evaluation_df = test_data.iloc[
            max(
                0,
                i - evaluation_window,
            ) : i
        ]

        X_eval = prepare_features(
            evaluation_df,
            features,
        )

        y_true = evaluation_df["Target"]

        y_pred = model.predict(X_eval)

        metrics = compute_classification_metrics(
            y_true,
            y_pred,
        )

        f1_scores.append(metrics["f1"])

        precision_scores.append(metrics["precision"])

        recall_scores.append(metrics["recall"])

        evaluation_steps.append(i)

        logger.info(
            f"Step={i} | "
            f"F1={metrics['f1']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )

    # FINAL SUMMARY
    if len(f1_scores) > 0:

        logger.info(f"Final F1: " f"{f1_scores[-1]:.4f}")

        logger.info(f"Average F1: " f"{np.mean(f1_scores):.4f}")

    logger.info(f"Total Drift Events: " f"{len(drift_events)}")

    logger.info(
        f"Retraining Frequency: "
        f"{len(drift_events)} / "
        f"{len(evaluation_steps)} "
        f"evaluation periods"
    )

    valid_drift_scores = [score for score in drift_scores if not np.isnan(score)]

    if valid_drift_scores:

        logger.info(f"Average Drift Score: " f"{np.mean(valid_drift_scores):.4f}")

        logger.info(f"Maximum Drift Score: " f"{np.max(valid_drift_scores):.4f}")

    # PERFORMANCE PLOT
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

    logger.info(f"Saved plot to {output_file}")

    logger.info("Drift retraining simulation completed.")


if __name__ == "__main__":
    simulate_drift_retraining()
