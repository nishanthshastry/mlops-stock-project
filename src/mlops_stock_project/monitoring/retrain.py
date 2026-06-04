import json
import shutil

import joblib
import pandas as pd

from mlops_stock_project.config import (
    MODEL_FILE,
    PROCESSED_DATA_FILE,
    PROJECT_ROOT,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

from mlops_stock_project.monitoring.drift import (
    detect_drift,
)

from mlops_stock_project.models.train_model import (
    train_and_track_models,
)

logger = get_logger(__name__)


# MONITORING OUTPUT

MONITORING_DIR = PROJECT_ROOT / "reports" / "monitoring"

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# RETRAINING PIPELINE


def retrain_if_drift_detected():

    logger.info("Starting retraining evaluation...")

    # LOAD DATA

    df = pd.read_csv(PROCESSED_DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Ticker", "Date"])

    logger.info(f"Loaded dataset shape: {df.shape}")

    # CREATE REFERENCE/CURRENT WINDOWS

    split_index = int(len(df) * 0.8)

    reference_df = df.iloc[:split_index].copy()

    current_df = df.iloc[split_index:].copy()

    logger.info(f"Reference rows: {len(reference_df)}")

    logger.info(f"Current rows: {len(current_df)}")

    # DETECT DRIFT

    drift_summary = detect_drift(
        reference_df,
        current_df,
    )

    drift_detected = drift_summary["overall_drift_detected"]

    # LOAD CURRENT MODEL

    current_artifact = joblib.load(MODEL_FILE)

    current_model_name = current_artifact.get(
        "model_name",
        "Unknown",
    )

    logger.info(f"Current model: {current_model_name}")

    retraining_triggered = False

    retraining_result = None

    # RETRAIN IF DRIFT DETECTED

    if drift_detected:
        logger.warning("Drift detected. Starting retraining...")

        retraining_triggered = True

        # Backup old model
        backup_path = MODEL_FILE.parent / "model_backup.pkl"

        shutil.copy(
            MODEL_FILE,
            backup_path,
        )

        logger.info(f"Backed up model to {backup_path}")

        # Retrain models
        best_model, best_metrics = train_and_track_models()

        retraining_result = best_metrics

        logger.info("Retraining completed.")

    else:
        logger.info("No retraining needed.")

    # SAVE REPORT

    report = {
        "drift_detected": bool(drift_detected),
        "retraining_triggered": bool(retraining_triggered),
        "current_model": str(current_model_name),
        "drift_summary": drift_summary,
        "retraining_result": retraining_result,
    }

    report_path = MONITORING_DIR / "retraining_report.json"

    with open(
        report_path,
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
            default=str,
        )

    logger.info(f"Retraining report saved to {report_path}")

    return report


# MAIN

if __name__ == "__main__":
    retrain_if_drift_detected()
