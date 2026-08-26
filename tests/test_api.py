# Este script verifica que la API responda HTTP 200 y devuelva la estructura correcta
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

# 1. Pruebas de Endpoints Básicos 
def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

# 2. Prueba de Inferencia Válida (input válido → predicción válida)
def test_predict_valid_input():
    payload = {
        "recency": 15,
        "frequency": 4,
        "monetary": 250.50,
        "product_diversity": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "cluster" in data
    assert isinstance(data["cluster"], int)

# 3. Pruebas de Validación de Datos (Tipos, Rangos y Faltantes)
def test_predict_invalid_data():
    # Falta el campo obligatorio 'monetary' o tiene tipo inválido
    payload = {"recency": "quince", "frequency": 4} 
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Error de validación Pydantic

# 4. Prueba del Endpoint de Monitoreo 
def test_drift_endpoint():
    payload = {
        "recent_data": [
            {"recency": 10, "frequency": 2, "monetary": 100.0, "product_diversity": 1}
        ]
    }
    response = client.post("/drift", json=payload)
    assert response.status_code == 200