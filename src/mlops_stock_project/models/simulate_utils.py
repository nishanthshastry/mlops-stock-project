import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier

from mlops_stock_project.config import (
    MODEL_FILE,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

logger = get_logger(__name__)


# LOAD MODEL ARTIFACT


def load_model_artifact():

    artifact = joblib.load(MODEL_FILE)

    return {
        "model": artifact["model"],
        "features": artifact["features"],
        "threshold": artifact.get(
            "threshold",
            0.5,
        ),
        "model_name": artifact.get(
            "model_name",
            "Unknown",
        ),
    }


# CREATE XGBOOST MODEL


def create_xgboost_model():

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )


# COMPUTE METRICS


def compute_classification_metrics(
    y_true,
    y_pred,
):

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
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

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# PLOT PERFORMANCE


def plot_simulation_metrics(
    evaluation_steps,
    f1_scores,
    precision_scores,
    recall_scores,
    output_file,
    title,
    vertical_markers=None,
):

    plt.figure(figsize=(12, 6))

    plt.plot(
        evaluation_steps,
        f1_scores,
        label="F1 Score",
    )

    plt.plot(
        evaluation_steps,
        precision_scores,
        label="Precision",
    )

    plt.plot(
        evaluation_steps,
        recall_scores,
        label="Recall",
    )

    # Optional markers
    if vertical_markers:
        for marker in vertical_markers:
            plt.axvline(
                x=marker,
                linestyle="--",
                alpha=0.5,
            )

    plt.xlabel("Evaluation Window")

    plt.ylabel("Metric Score")

    plt.title(title)

    plt.legend()

    plt.grid(True)

    plt.savefig(
        output_file,
        bbox_inches="tight",
        dpi=300,
    )

    plt.close()

    logger.info(f"Saved performance plot to {output_file}")
