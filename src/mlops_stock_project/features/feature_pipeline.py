import numpy as np
import pandas as pd

from mlops_stock_project.logging_config import (
    get_logger,
)

logger = get_logger(__name__)


# RECREATE TICKER DUMMIES


def recreate_ticker_dummies(
    df,
):
    """
    Recreate one-hot ticker columns
    used during training.
    """

    if "Ticker" in df.columns:
        logger.info("Recreating ticker dummy variables...")

        ticker_dummies = pd.get_dummies(
            df["Ticker"],
            prefix="Ticker",
        )

        df = pd.concat(
            [df, ticker_dummies],
            axis=1,
        )

    return df


# ALIGN FEATURES


def align_features(
    df,
    features,
):
    """
    Ensure all expected model
    features exist.
    """

    missing_features = [feature for feature in features if feature not in df.columns]

    if missing_features:
        logger.warning(f"Missing features detected: {missing_features}")

        for feature in missing_features:
            df[feature] = 0

    return df


# CONVERT FEATURES TO NUMERIC


def convert_numeric_features(
    X,
):
    """
    Convert all features into
    stable numeric float32 values.
    """

    # Convert booleans
    bool_columns = X.select_dtypes(include=["bool"]).columns

    if len(bool_columns) > 0:
        X[bool_columns] = X[bool_columns].astype(int)

    # Force numeric conversion
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Replace infinities
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Fill NaNs
    X = X.fillna(0)

    # Final dtype
    X = X.astype("float32")

    return X


# PREPARE FEATURES


def prepare_features(
    df,
    features,
):
    """
    Full production-safe
    feature preparation pipeline.
    """

    logger.info("Preparing feature matrix...")

    # Recreate ticker dummies
    df = recreate_ticker_dummies(df)

    # Align expected features
    df = align_features(
        df,
        features,
    )

    # Feature matrix
    X = df.reindex(
        columns=features,
        fill_value=0,
    ).copy()

    # Numeric conversion
    X = convert_numeric_features(X)

    logger.info(f"Prepared feature matrix: {X.shape}")

    return X
