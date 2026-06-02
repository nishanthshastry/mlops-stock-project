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


class StockFeatures(BaseModel):
    Return: float
    MA_5: float
    MA_10: float
    Volatility: float


# Load trained model
model = joblib.load(MODEL_FILE)

logger.info(f"Loaded model from {MODEL_FILE}")


@app.get("/")
def home():
    return {"message": "MLOps Stock Prediction API Running"}


@app.post("/predict")
def predict(data: StockFeatures):
    try:
        df = pd.DataFrame([data.model_dump()])

        prediction = model.predict(df)[0]

        logger.info(f"Prediction made: {prediction}")

        return {"prediction": int(prediction)}

    except Exception as e:
        logger.error(str(e))
        return {"error": str(e)}

@app.post("/retrain")
def retrain():

    try:

        retrain_if_drift_detected()

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