import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.pipeline import Pipeline

from mlops_stock_project.config import (
    MODEL_FILE,
)

from mlops_stock_project.features.feature_pipeline import (
    prepare_features,
)

from mlops_stock_project.logging_config import (
    get_logger,
)

from mlops_stock_project.monitoring.retrain import (
    retrain_if_drift_detected,
)

logger = get_logger(__name__)


# FASTAPI APP

app = FastAPI(
    title="MLOps Stock Prediction API",
    version="2.0.0",
)


# GLOBAL ARTIFACTS

model = None

threshold = 0.5

features = []

model_name = "Unknown"


# REQUEST SCHEMA


class StockFeatures(BaseModel):
    Return: float

    MA_5: float

    MA_10: float

    Volatility: float

    Lag_1: float

    Lag_2: float

    Lag_3: float

    Momentum_5: float

    EMA_10: float

    RSI: float

    Volume_Change: float

    Volume_MA_5: float

    MACD: float

    BB_upper: float

    BB_lower: float

    SPY_Return: float = 0

    SPY_MA_5: float = 0

    SPY_Volatility: float = 0

    QQQ_Return: float = 0

    QQQ_Momentum: float = 0

    QQQ_MA_10: float = 0

    VIX_Return: float = 0

    VIX_MA_5: float = 0

    VIX_Level: float = 20

    High_VIX_Regime: int = 0

    Relative_SPY_Strength: float = 0

    Relative_QQQ_Strength: float = 0

    Market_Stress: int = 0

    # Optional ticker flags
    Ticker_AAPL: int = 0

    Ticker_AMD: int = 0

    Ticker_AMZN: int = 0

    Ticker_GOOGL: int = 0

    Ticker_INTC: int = 0

    Ticker_META: int = 0

    Ticker_MSFT: int = 0

    Ticker_NFLX: int = 0

    Ticker_NVDA: int = 0

    Ticker_TSLA: int = 0


# LOAD MODEL


def load_model():

    global model

    global threshold

    global features

    global model_name

    try:
        if not os.path.exists(MODEL_FILE):
            logger.error(f"Model file not found: {MODEL_FILE}")

            model = None

            return

        artifact = joblib.load(MODEL_FILE)

        # Metadata artifact
        if isinstance(
            artifact,
            dict,
        ):
            model = artifact.get("model")

            threshold = artifact.get(
                "threshold",
                0.5,
            )

            features = artifact.get(
                "features",
                [],
            )

            model_name = artifact.get(
                "model_name",
                "Unknown",
            )

        else:
            # Legacy support
            model = artifact

            threshold = 0.5

            features = []

            model_name = "LegacyModel"

        logger.info(f"Loaded model: {model_name}")

        logger.info(f"Threshold: {threshold:.4f}")

        logger.info(f"Feature count: {len(features)}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

        model = None


# Load at startup
load_model()


# ROOT ENDPOINT


@app.get("/")
def home():

    return {
        "message": "MLOps Stock Prediction API Running",
        "model_loaded": model is not None,
        "model_name": model_name,
        "threshold": round(
            threshold,
            4,
        ),
        "feature_count": len(features),
    }


# HEALTH CHECK


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name,
    }


# PREDICTION ENDPOINT


@app.post("/predict")
def predict(
    data: StockFeatures,
):

    try:
        if model is None:
            raise HTTPException(
                status_code=500,
                detail=("Model not loaded"),
            )

        # CONVERT REQUEST

        df = pd.DataFrame([data.model_dump()])

        # FEATURE PREPARATION

        X = prepare_features(
            df,
            features,
        )

        # HANDLE PIPELINES

        actual_model = model

        if isinstance(
            model,
            Pipeline,
        ):
            scaler = model.named_steps["scaler"]

            X_transformed = scaler.transform(X)

            actual_model = model.named_steps["model"]

        else:
            X_transformed = X

        # PREDICT

        probability = actual_model.predict_proba(X_transformed)[0][1]

        prediction = int(probability >= threshold)

        # CONFIDENCE

        confidence_score = abs(probability - threshold)

        if confidence_score >= 0.30:
            confidence_label = "high"

        elif confidence_score >= 0.15:
            confidence_label = "medium"

        else:
            confidence_label = "low"

        logger.info(f"Prediction={prediction} | Probability={probability:.4f}")

        return {
            "prediction": prediction,
            "probability": round(
                float(probability),
                4,
            ),
            "threshold": round(
                float(threshold),
                4,
            ),
            "confidence": confidence_label,
            "confidence_score": round(
                float(confidence_score),
                4,
            ),
            "model_name": model_name,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# MANUAL RETRAINING


@app.post("/retrain")
def retrain():

    try:
        result = retrain_if_drift_detected()

        # Reload updated model
        load_model()

        return {
            "status": "retraining_completed",
            "details": result,
        }

    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
