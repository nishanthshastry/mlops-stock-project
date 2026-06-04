from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_classification_model(
    y_true,
    y_pred,
):

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    # Optional ROC-AUC
    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_pred,
        )

    except Exception:
        metrics["roc_auc"] = 0.0

    return metrics
