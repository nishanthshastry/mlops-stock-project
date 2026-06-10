from fastapi.testclient import (
    TestClient,
)

from mlops_stock_project.api.app import (
    app,
)

client = TestClient(app)


# HOME ENDPOINT
def test_home():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data

    assert "model_loaded" in data

    assert "model_name" in data

    assert "feature_count" in data

    assert "threshold" in data

    assert data["model_loaded"] is True


# HEALTH ENDPOINT
def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data

    assert data["status"] == "healthy"

    assert "model_loaded" in data

    assert "model_name" in data


# PREDICTION ENDPOINT
def test_predict():

    payload = {
        # Core Features
        "Return": 0.01,
        "MA_5": 170.0,
        "MA_10": 168.0,
        "Volatility": 2.5,
        "Lag_1": 0.005,
        "Lag_2": 0.002,
        "Lag_3": -0.001,
        "Momentum_5": 0.03,
        "EMA_10": 169.5,
        "RSI": 58.0,
        "Volume_Change": 0.12,
        "Volume_MA_5": 1000000.0,
        "MACD": 1.5,
        "BB_upper": 175.0,
        "BB_lower": 165.0,
        # Market Features
        "SPY_Return": 0.004,
        "SPY_MA_5": 520.0,
        "SPY_Volatility": 1.2,
        "QQQ_Return": 0.006,
        "QQQ_Momentum": 0.02,
        "QQQ_MA_10": 440.0,
        "VIX_Return": -0.01,
        "VIX_MA_5": 14.0,
        "VIX_Level": 13.5,
        "High_VIX_Regime": 0,
        # Relative Market Features
        "Relative_SPY_Strength": 0.007,
        "Relative_QQQ_Strength": 0.009,
        "Relative_SPY_Volatility": 0.10,
        "Relative_VIX_Level": -0.05,
        # Regime Features
        "Market_Stress": 0,
        "Sector_Strength": 0.012,
        # Sector Features
        "Sector_Technology": 1,
        "Sector_Healthcare": 0,
        "Sector_Financials": 0,
        "Sector_Consumer": 0,
        "Sector_Energy": 0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "probability" in data

    assert "threshold" in data

    assert "confidence" in data

    assert "confidence_score" in data

    assert "model_name" in data

    assert isinstance(
        data["prediction"],
        int,
    )

    assert isinstance(
        data["probability"],
        float,
    )

    assert isinstance(
        data["confidence_score"],
        float,
    )

    assert data["confidence"] in [
        "low",
        "medium",
        "high",
    ]
