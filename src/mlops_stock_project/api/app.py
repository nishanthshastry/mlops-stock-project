import os

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from mlops_stock_project.config import MODEL_FILE
from mlops_stock_project.logging_config import get_logger
from mlops_stock_project.monitoring.retrain import (
    retrain_if_drift_detected,
)

logger = get_logger(__name__)

app = FastAPI(title="MLOps Stock Prediction API")

model = None


class StockFeatures(BaseModel):
    Return: float
    MA_5: float
    MA_10: float
    Volatility: float


def load_model():
    """
    Load trained model safely.
    """

    global model

    try:

        if os.path.exists(MODEL_FILE):

            model = joblib.load(MODEL_FILE)

            logger.info(
                f"Loaded model from {MODEL_FILE}"
            )

        else:

            logger.warning(
                f"Model file not found at {MODEL_FILE}"
            )

            model = None

    except Exception as e:

        logger.error(
            f"Error loading model: {str(e)}"
        )

        model = None


# Load model during startup
load_model()


@app.get("/")
def home():

    return {
        "message": (
            "MLOps Stock Prediction API Running"
        )
    }


@app.post("/predict")
def predict(data: StockFeatures):

    try:

        if model is None:

            return {
                "error": (
                    "Model not loaded. "
                    "Please train or pull model artifacts."
                )
            }

        df = pd.DataFrame(
            [data.model_dump()]
        )

        prediction = model.predict(df)[0]

        logger.info(
            f"Prediction made: {prediction}"
        )

        return {
            "prediction": int(prediction)
        }

    except Exception as e:

        logger.error(str(e))

        return {
            "error": str(e)
        }


@app.post("/retrain")
def retrain():

    try:

        retrain_if_drift_detected()

        # Reload latest model after retraining
        load_model()

        return {
            "message": (
                "Retraining workflow completed."
            )
        }

    except Exception as e:

        logger.error(str(e))

        return {
            "error": str(e)
        }