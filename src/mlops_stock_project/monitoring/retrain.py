from mlops_stock_project.logging_config import get_logger
from mlops_stock_project.monitoring.drift import detect_drift
from mlops_stock_project.models.train_model import train_model

logger = get_logger(__name__)


def retrain_if_drift_detected():

    logger.info("Checking for drift...")

    drift_detected = detect_drift()

    if drift_detected:

        logger.warning(
            "Drift detected. Retraining model..."
        )

        model, metrics = train_model()

        logger.info(
            f"Retraining complete. Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

    else:

        logger.info(
            "No drift detected. Retraining skipped."
        )


if __name__ == "__main__":
    retrain_if_drift_detected()