import pandas as pd

from mlops_stock_project.logging_config import get_logger
from mlops_stock_project.config import PROCESSED_DATA_FILE

logger = get_logger(__name__)


def detect_drift(
    reference_data_path=PROCESSED_DATA_FILE,
    new_data_path=PROCESSED_DATA_FILE,
    threshold=0.1,
):
    """
    Simple drift detection using mean percentage difference.
    """

    logger.info("Starting drift detection...")

    reference_df = pd.read_csv(reference_data_path)
    new_df = pd.read_csv(new_data_path).copy()

    # Simulate drift
    new_df["Volatility"] = new_df["Volatility"] * 2.5
    new_df["Return"] = new_df["Return"] * 1.8

    features = ["Return", "MA_5", "MA_10", "Volatility"]

    drift_detected = False

    for feature in features:

        reference_mean = reference_df[feature].mean()
        new_mean = new_df[feature].mean()

        if reference_mean == 0:
            continue

        drift_score = abs(new_mean - reference_mean) / abs(reference_mean)

        logger.info(
            f"{feature} drift score: {drift_score:.4f}"
        )

        if drift_score > threshold:
            logger.warning(
                f"Drift detected in feature: {feature}"
            )
            drift_detected = True

    if drift_detected:
        logger.warning("Overall data drift detected.")
    else:
        logger.info("No significant drift detected.")

    return drift_detected


if __name__ == "__main__":
    detect_drift()