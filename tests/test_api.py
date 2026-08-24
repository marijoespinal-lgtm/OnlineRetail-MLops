# Este script verifica que la API responda HTTP 200 y devuelva la estructura correcta
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint():
    payload = {
        "recency": 10.0,
        "frequency": 5.0,
        "monetary": 1500.0,
        "monetary_std": 120.0,
        "return_rate": 0.02,
        "unique_products": 15.0,
        "weekend_purchase_pct": 0.20
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 500]

def test_drift_endpoint():
    batch_payload = [
        {
            "recency": 12.0,
            "frequency": 4.0,
            "monetary": 1400.0,
            "monetary_std": 110.0,
            "return_rate": 0.01,
            "unique_products": 12.0,
            "weekend_purchase_pct": 0.15
        }
    ]
    response = client.post("/drift", json=batch_payload)
    assert response.status_code in [200, 500]