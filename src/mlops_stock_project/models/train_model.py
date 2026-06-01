import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression

from mlops_stock_project.config import (
    PROCESSED_DATA_FILE,
    MODEL_DIR,
    MODEL_FILE
)

from mlops_stock_project.logging_config import get_logger

from mlops_stock_project.evaluation.metrics import (
    evaluate_classification_model
)

logger = get_logger(__name__)


def train_model(data_path=PROCESSED_DATA_FILE):
    logger.info("Training model...")

    df = pd.read_csv(data_path)

    features = [
        "Return",
        "MA_5",
        "MA_10",
        "Volatility"
    ]

    X = df[features]
    y = df["Target"]

    # Time-based split
    split_index = int(len(df) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    # Train model
    model = LogisticRegression()

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = evaluate_classification_model(
        y_test,
        y_pred
    )

    logger.info(
        f"Model Accuracy: {metrics['accuracy']:.4f}"
    )

    # Save model artifact
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    logger.info(f"Model saved to {MODEL_FILE}")

    return model, metrics


if __name__ == "__main__":
    train_model()