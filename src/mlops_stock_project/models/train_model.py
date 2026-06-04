import os
import sys
import subprocess

import joblib
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from sklearn.model_selection import (
    TimeSeriesSplit,
)

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier

from mlops_stock_project.config import (
    MODEL_DIR,
    MODEL_FILE,
    PROCESSED_DATA_FILE,
    PROJECT_ROOT,
    REPORTS_FIGURES_DIR,
)

from mlops_stock_project.features.feature_pipeline import (
    prepare_features,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

logger = get_logger(__name__)


# THRESHOLD OPTIMIZATION


def optimize_threshold(
    y_true,
    y_prob,
):

    best_threshold = 0.5

    best_f1 = 0

    thresholds = np.arange(
        0.30,
        0.71,
        0.02,
    )

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        score = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score

            best_threshold = threshold

    return (
        best_threshold,
        best_f1,
    )


# MODEL DEFINITIONS


def build_models():

    return {
        "LogisticRegression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=0.5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                min_samples_split=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=2,
                min_samples_split=4,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=250,
                max_depth=8,
                learning_rate=0.02,
                subsample=0.85,
                colsample_bytree=0.85,
                gamma=1,
                min_child_weight=3,
                reg_alpha=0.5,
                reg_lambda=1.5,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
        ),
    }


# MAIN TRAINING FUNCTION


def train_and_track_models(
    data_path=PROCESSED_DATA_FILE,
):

    logger.info("Training models with TimeSeriesSplit...")

    # LOAD DATA

    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Ticker", "Date"])

    logger.info(f"Dataset rows: {len(df)}")

    # FEATURE LIST

    features = [
        # Core features
        "Return",
        "MA_5",
        "MA_10",
        "Volatility",
        "Lag_1",
        "Lag_2",
        "Lag_3",
        "Momentum_5",
        "EMA_10",
        "RSI",
        "Volume_Change",
        "Volume_MA_5",
        "MACD",
        "BB_upper",
        "BB_lower",
        # Market context
        "SPY_Return",
        "SPY_MA_5",
        "SPY_Volatility",
        "QQQ_Return",
        "QQQ_Momentum",
        "QQQ_MA_10",
        "VIX_Return",
        "VIX_MA_5",
        "VIX_Level",
        "High_VIX_Regime",
        "Relative_SPY_Strength",
        "Relative_QQQ_Strength",
        "Market_Stress",
        # Ticker dummies
        "Ticker_AAPL",
        "Ticker_AMD",
        "Ticker_AMZN",
        "Ticker_GOOGL",
        "Ticker_INTC",
        "Ticker_META",
        "Ticker_MSFT",
        "Ticker_NFLX",
        "Ticker_NVDA",
        "Ticker_TSLA",
    ]

    # PREPARE FEATURES

    X = prepare_features(
        df,
        features,
    )

    y = df["Target"]

    logger.info(f"Using {len(features)} features")

    # MLFLOW SETUP

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")

    mlflow.set_experiment("mlops-stock-prediction")

    # TIMESERIES CV

    tscv = TimeSeriesSplit(
        n_splits=3,
    )

    models = build_models()

    # BEST MODEL TRACKING

    best_model = None

    best_model_name = ""

    best_threshold = 0.5

    best_score = 0

    best_metrics = {}

    # TRAINING LOOP

    for (
        model_name,
        model,
    ) in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"\nTraining {model_name}")

            fold_scores = []

            fold_thresholds = []

            final_y_test = None

            final_y_pred = None

            # WALK-FORWARD VALIDATION

            for (
                fold,
                (
                    train_idx,
                    test_idx,
                ),
            ) in enumerate(tscv.split(X)):
                logger.info(f"{model_name} - Fold {fold + 1}")

                X_train = X.iloc[train_idx]

                X_test = X.iloc[test_idx]

                y_train = y.iloc[train_idx]

                y_test = y.iloc[test_idx]

                # TRAIN
                model.fit(
                    X_train,
                    y_train,
                )

                # PREDICT PROBA
                y_prob = model.predict_proba(X_test)[:, 1]

                # OPTIMIZE THRESHOLD
                threshold, fold_f1 = optimize_threshold(
                    y_test,
                    y_prob,
                )

                y_pred = (y_prob >= threshold).astype(int)

                fold_scores.append(fold_f1)

                fold_thresholds.append(threshold)

                logger.info(f"Fold {fold + 1} F1: {fold_f1:.4f}")

                # SAVE LAST FOLD
                final_y_test = y_test

                final_y_pred = y_pred

            # FINAL METRICS

            avg_f1 = np.mean(fold_scores)

            avg_threshold = np.mean(fold_thresholds)

            precision = precision_score(
                final_y_test,
                final_y_pred,
                zero_division=0,
            )

            recall = recall_score(
                final_y_test,
                final_y_pred,
                zero_division=0,
            )

            logger.info(f"{model_name} Average F1: {avg_f1:.4f}")

            # MLFLOW LOGGING

            mlflow.log_param(
                "model_type",
                model_name,
            )

            mlflow.log_metric(
                "avg_f1_score",
                avg_f1,
            )

            mlflow.log_metric(
                "precision",
                precision,
            )

            mlflow.log_metric(
                "recall",
                recall,
            )

            mlflow.log_metric(
                "optimal_threshold",
                avg_threshold,
            )

            # FEATURE IMPORTANCE

            actual_model = model

            if isinstance(
                model,
                Pipeline,
            ):
                actual_model = model.named_steps["model"]

            if hasattr(
                actual_model,
                "feature_importances_",
            ):
                importance_df = pd.DataFrame(
                    {
                        "feature": features,
                        "importance": actual_model.feature_importances_,
                    }
                ).sort_values(
                    by="importance",
                    ascending=False,
                )

                logger.info(f"\nTop Features:\n{importance_df.head(15)}")

            # CONFUSION MATRIX

            os.makedirs(
                REPORTS_FIGURES_DIR,
                exist_ok=True,
            )

            fig_path = REPORTS_FIGURES_DIR / f"{model_name}_cm.png"

            ConfusionMatrixDisplay.from_predictions(
                final_y_test,
                final_y_pred,
            )

            plt.savefig(fig_path)

            plt.close()

            mlflow.log_artifact(fig_path)

            # LOG MODEL

            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
            )

            # BEST MODEL

            if avg_f1 > best_score:
                best_score = avg_f1

                best_model = model

                best_model_name = model_name

                best_threshold = avg_threshold

                best_metrics = {
                    "f1_score": 0.0 if np.isnan(avg_f1) else float(avg_f1),
                    "precision": 0.0 if np.isnan(precision) else float(precision),
                    "recall": 0.0 if np.isnan(recall) else float(recall),
                    "threshold": (
                        0.5 if np.isnan(avg_threshold) else float(avg_threshold)
                    ),
                }

    # SAVE BEST MODEL

    logger.info(f"\nBest Model: {best_model_name}")

    logger.info(f"Best Average F1: {best_score:.4f}")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "threshold": best_threshold,
            "features": features,
        },
        MODEL_FILE,
    )

    logger.info(f"Best model saved to {MODEL_FILE}")

    # DVC TRACKING

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "dvc",
                "add",
                str(MODEL_FILE),
            ],
            check=True,
        )

        logger.info("Model tracked with DVC")

    except Exception as e:
        logger.warning(f"DVC tracking failed: {str(e)}")

    return (
        best_model,
        best_metrics,
    )


# BACKWARD COMPATIBILITY


def train_model(
    data_path=PROCESSED_DATA_FILE,
):

    return train_and_track_models(data_path)


# MAIN

if __name__ == "__main__":
    train_and_track_models()
