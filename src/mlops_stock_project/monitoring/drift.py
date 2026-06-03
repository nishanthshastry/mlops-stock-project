import json

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from mlops_stock_project.config import (
    PROCESSED_DATA_FILE,
    PROJECT_ROOT,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

logger = get_logger(__name__)


# MONITORING CONFIG

MONITORING_DIR = PROJECT_ROOT / "reports" / "monitoring"

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# FEATURES TO MONITOR

MONITORED_FEATURES = [
    # Core stock features
    "Return",
    "Volatility",
    "RSI",
    "MACD",
    # Market context
    "SPY_Return",
    "QQQ_Return",
    "VIX_Level",
    # Relative strength
    "Relative_SPY_Strength",
    # Regime feature
    "Market_Stress",
]


# PSI CALCULATION


def calculate_psi(
    expected,
    actual,
    buckets=10,
):
    """
    Calculate Population Stability Index.
    """

    expected = np.array(
        expected,
        dtype=np.float64,
    )

    actual = np.array(
        actual,
        dtype=np.float64,
    )

    # Remove NaNs
    expected = expected[~np.isnan(expected)]

    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Percentile bins
    breakpoints = np.percentile(
        expected,
        np.arange(
            0,
            buckets + 1,
        )
        / buckets
        * 100,
    )

    breakpoints = np.unique(breakpoints)

    # Prevent invalid bins
    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(
        expected,
        bins=breakpoints,
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=breakpoints,
    )[0]

    expected_percents = expected_counts / len(expected)

    actual_percents = actual_counts / len(actual)

    # Avoid divide-by-zero
    expected_percents = np.where(
        expected_percents == 0,
        0.0001,
        expected_percents,
    )

    actual_percents = np.where(
        actual_percents == 0,
        0.0001,
        actual_percents,
    )

    psi = np.sum(
        (actual_percents - expected_percents)
        * np.log(actual_percents / expected_percents)
    )

    return round(
        float(psi),
        4,
    )


# DRIFT SEVERITY


def classify_drift_severity(
    psi_score,
):
    """
    PSI interpretation.
    """

    if psi_score < 0.1:
        return "stable"

    elif psi_score < 0.25:
        return "moderate"

    return "significant"


# MAIN DRIFT DETECTION


def detect_drift(
    reference_df,
    current_df,
    psi_threshold=0.25,
    ks_threshold=0.05,
):
    """
    Detect drift between two datasets.
    """

    logger.info("Starting advanced drift detection...")

    logger.info(f"Reference rows: {len(reference_df)}")

    logger.info(f"Current rows: {len(current_df)}")

    # DRIFT STORAGE

    drift_detected = False

    drift_results = {}

    # FEATURE MONITORING

    for feature in MONITORED_FEATURES:
        try:
            # Skip missing features
            if feature not in reference_df.columns or feature not in current_df.columns:
                logger.warning(f"Skipping missing feature: {feature}")

                continue

            expected = reference_df[feature].dropna().astype("float32")

            actual = current_df[feature].dropna().astype("float32")

            # PSI
            psi_score = calculate_psi(
                expected,
                actual,
            )

            # KS Test
            ks_statistic, ks_pvalue = ks_2samp(
                expected,
                actual,
            )

            severity = classify_drift_severity(psi_score)

            # FINAL DRIFT DECISION
            feature_drift = (psi_score > psi_threshold) and (ks_pvalue < ks_threshold)

            if feature_drift:
                drift_detected = True

            # SAVE RESULT
            drift_results[feature] = {
                "psi_score": round(
                    float(psi_score),
                    4,
                ),
                "ks_pvalue": round(
                    float(ks_pvalue),
                    6,
                ),
                "severity": str(severity),
                "drift_detected": bool(feature_drift),
            }

            # LOGGING
            logger.info(
                f"{feature} | "
                f"PSI="
                f"{psi_score:.4f} | "
                f"KS p-value="
                f"{ks_pvalue:.6f} | "
                f"Severity="
                f"{severity}"
            )

            if feature_drift:
                logger.warning(f"Drift detected in {feature}")

        except Exception as e:
            logger.warning(f"Failed monitoring {feature}: {str(e)}")

    # SUMMARY

    drifted_features = [
        feature
        for (
            feature,
            result,
        ) in drift_results.items()
        if result["drift_detected"]
    ]

    summary = {
        "overall_drift_detected": bool(drift_detected),
        "drifted_features": [str(feature) for feature in drifted_features],
        "feature_results": {},
    }

    # Safe JSON conversion
    for (
        feature,
        result,
    ) in drift_results.items():
        summary["feature_results"][str(feature)] = {
            "psi_score": float(result["psi_score"]),
            "ks_pvalue": float(result["ks_pvalue"]),
            "severity": str(result["severity"]),
            "drift_detected": bool(result["drift_detected"]),
        }

    # SAVE REPORT

    report_path = MONITORING_DIR / "drift_report.json"

    with open(
        report_path,
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    logger.info(f"Drift report saved to {report_path}")

    # FINAL LOGGING

    if drift_detected:
        logger.warning("Overall drift detected.")

        logger.warning(f"Drifted features: {drifted_features}")

    else:
        logger.info("No significant drift detected.")

    return summary


# STANDALONE EXECUTION

if __name__ == "__main__":
    logger.info("Running standalone drift monitoring...")

    df = pd.read_csv(PROCESSED_DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Ticker", "Date"])

    split_index = int(len(df) * 0.8)

    reference_df = df.iloc[:split_index].copy()

    current_df = df.iloc[split_index:].copy()

    detect_drift(
        reference_df,
        current_df,
    )
