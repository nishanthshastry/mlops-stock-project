from fastapi.testclient import TestClient

from mlops_stock_project.api.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "MLOps Stock Prediction API Running"}


def test_predict():
    payload = {"Return": 0.01, "MA_5": 170, "MA_10": 168, "Volatility": 2.5}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in [0, 1]
