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


# SCHEDULED RETRAINING


def simulate_retraining(
    data_path=PROCESSED_DATA_FILE,
):
    logger.info("Running scheduled retraining simulation...")

    # LOAD DATA

    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        [
            "Ticker",
            "Date",
        ]
    )

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

    model = create_xgboost_model(
        fast_mode=True,
    )

    model.fit(
        X_train,
        y_train,
    )

    # CONFIG

    window_size = 500

    retrain_interval = 1000

    f1_scores = []

    precision_scores = []

    recall_scores = []

    evaluation_steps = []

    retrain_points = []

    logger.info(f"Evaluating {len(test_data)} test rows in chunks of {window_size}")

    # FAST EVALUATION LOOP

    for i in range(
        window_size,
        len(test_data),
        window_size,
    ):
        # RETRAIN

        if i % retrain_interval == 0:
            logger.info(f"Retraining model at step {i}")

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

            retrain_points.append(i)

        # EVALUATE

        evaluation_df = test_data.iloc[:i]

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

        logger.info(f"Step={i} | F1={metrics['f1']:.4f}")

    if len(f1_scores) > 0:
        logger.info(f"Final F1: {f1_scores[-1]:.4f}")

    # PLOT

    output_file = REPORTS_FIGURES_DIR / "scheduled_retraining_performance.png"

    plot_simulation_metrics(
        evaluation_steps=evaluation_steps,
        f1_scores=f1_scores,
        precision_scores=precision_scores,
        recall_scores=recall_scores,
        output_file=output_file,
        title="Scheduled Retraining Performance",
        vertical_markers=retrain_points,
    )

    logger.info("Scheduled retraining simulation completed.")


if __name__ == "__main__":
    simulate_retraining()
