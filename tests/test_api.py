# Este script verifica que la API responda HTTP 200 y devuelva la estructura correcta
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_check():
    """Prueba que el endpoint de estado responda HTTP 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint_valid():
    """Prueba de entrada válida a la API."""
    payload = {
        "customer_id": "12345",
        "recency": 10.5,
        "frequency": 3,
        "monetary": 250.0,
        "weekend_ratio": 0.2,
        "product_diversity": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "assigned_cluster" in data
    assert "model_version" in data
    assert data["customer_id"] == "12345"

def test_predict_endpoint_invalid_data():
    """Prueba de comportamiento frente a un input inválido (error de validación)."""
    payload = {
        "customer_id": "12345",
        "recency": "diez_dias",  # Debería ser numérico
        "frequency": "muchas"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity