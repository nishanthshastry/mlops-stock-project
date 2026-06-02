import os
import subprocess
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
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

from sklearn.metrics import (
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier

from mlops_stock_project.config import (
    MODEL_DIR,
    MODEL_FILE,
    PROCESSED_DATA_FILE,
    PROJECT_ROOT,
    FIGURES_DIR,
)

from mlops_stock_project.evaluation.metrics import (
    evaluate_classification_model,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

logger = get_logger(__name__)


def train_model(data_path=PROCESSED_DATA_FILE,):

    logger.info("Training model...")

    # Load Dataset
    df = pd.read_csv(data_path)

    # Sort chronologically
    df = df.sort_values(
        ["Date", "Ticker"]
    )

    # Encode Ticker
    ticker_dummies = pd.get_dummies(
        df["Ticker"],
        prefix="Ticker",
    )

    df = pd.concat(
        [df, ticker_dummies],
        axis=1,
    )

    
    # Features
    base_features = [
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
    ]

    ticker_features = list(
        ticker_dummies.columns
    )

    features = (
        base_features
        + ticker_features
    )

    X = df[features]

    y = df["Target"]

    
    # Chronological Split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]

    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]

    y_test = y.iloc[split_index:]

    logger.info(
        f"Training rows: {len(X_train)}"
    )

    logger.info(
        f"Testing rows: {len(X_test)}"
    )

    
    # MLflow Setup
    mlflow.set_tracking_uri(
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        "mlops-stock-prediction"
    )

    
    # Models
    models = {

        "LogisticRegression": Pipeline([
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    }

    
    # Best Model Tracking
    best_model = None

    best_metrics = None

    best_score = 0

    best_model_name = ""

    
    # Train + Compare
    for model_name, model in models.items():

        with mlflow.start_run(
            run_name=model_name
        ):

            logger.info(
                f"Training {model_name}"
            )

            # Train
            model.fit(
                X_train,
                y_train,
            )

            # Predict
            y_pred = model.predict(
                X_test
            )

            # Metrics
            metrics = (
                evaluate_classification_model(
                    y_test,
                    y_pred,
                )
            )

            score = metrics["f1_score"]

            logger.info(
                f"{model_name} "
                f"F1 Score: {score:.4f}"
            )

            
            # MLflow Logging
            mlflow.log_param(
                "model_type",
                model_name,
            )

            mlflow.log_metric(
                "f1_score",
                score,
            )

            for (
                metric_name,
                metric_value,
            ) in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value,
                )

            
            # Feature Importance
            if hasattr(
                model,
                "feature_importances_",
            ):

                importance_df = (
                    pd.DataFrame({
                        "feature": features,
                        "importance": (
                            model.feature_importances_
                        ),
                    })
                    .sort_values(
                        by="importance",
                        ascending=False,
                    )
                )

                logger.info(
                    f"\nFeature Importance "
                    f"for {model_name}:\n"
                    f"{importance_df.head(15)}"
                )

            
            # Confusion Matrix
            os.makedirs(
                FIGURES_DIR,
                exist_ok=True,
            )

            fig_path = (
                FIGURES_DIR
                / f"{model_name}_cm.png"
            )

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                y_pred,
            )

            plt.savefig(fig_path)

            plt.close()

            mlflow.log_artifact(
                fig_path
            )

            
            # Log Model
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
            )

            
            # Best Model Selection
            if score > best_score:

                best_score = score

                best_model = model

                best_metrics = metrics

                best_model_name = model_name

    
    # Save Best Model
    logger.info(
        f"Best model: {best_model_name}"
    )

    logger.info(
        f"Best F1 Score: "
        f"{best_score:.4f}"
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_FILE,
    )

    logger.info(
        f"Best model saved to "
        f"{MODEL_FILE}"
    )

    
    # Track with DVC
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

        logger.info(
            "Model tracked with DVC"
        )

    except Exception as e:

        logger.warning(
            f"DVC tracking failed: "
            f"{str(e)}"
        )

    return best_model, best_metrics


if __name__ == "__main__":

    train_model()