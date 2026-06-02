import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression

from mlops_stock_project.config import (
    MODEL_DIR,
    MODEL_FILE,
    PROCESSED_DATA_FILE,
)
from mlops_stock_project.evaluation.metrics import (
    evaluate_classification_model,
)
from mlops_stock_project.logging_config import get_logger

logger = get_logger(__name__)


def train_model(data_path=PROCESSED_DATA_FILE):
    logger.info("Training model...")

    # Load processed dataset
    df = pd.read_csv(data_path)

    features = ["Return", "MA_5", "MA_10", "Volatility"]

    X = df[features]
    y = df["Target"]

    # Time-based split
    split_index = int(len(df) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    # MLflow experiment tracking
    mlflow.set_experiment("mlops-stock-prediction")

    with mlflow.start_run():

        logger.info("Starting MLflow run...")

        # Model parameters
        max_iter = 200
        random_state = 42

        # Train model
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )

        model.fit(X_train, y_train)

        logger.info("Model training completed")

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = evaluate_classification_model(y_test, y_pred)

        accuracy = metrics["accuracy"]

        logger.info(f"Model Accuracy: {accuracy:.4f}")

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("random_state", random_state)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Save model artifact locally
        os.makedirs(MODEL_DIR, exist_ok=True)

        joblib.dump(model, MODEL_FILE)

        logger.info(f"Model saved to {MODEL_FILE}")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        # Log model artifact file
        mlflow.log_artifact(MODEL_FILE)

        logger.info("MLflow tracking completed")

    return model, metrics


if __name__ == "__main__":
    train_model()