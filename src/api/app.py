from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Online Retail Customer Segmentation API",
    description="API para predicción de clústeres de clientes y monitoreo de Data Drift (PSI)",
    version="1.0.0"
)

# Definición del esquema de entrada según feature_schema.json
class CustomerData(BaseModel):
    recency: float
    frequency: float
    monetary: float
    monetary_std: float
    return_rate: float
    unique_products: float
    weekend_purchase_pct: float

# Rutas de los artefactos del modelo y dataset baseline
MODEL_PATH = "models/customer_segmentation_model.pkl"
SCALER_PATH = "models/scaler.pkl"
BASELINE_PATH = "data/processed/customer_features.csv"

model = None
scaler = None
baseline_df = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

if os.path.exists(BASELINE_PATH):
    baseline_df = pd.read_csv(BASELINE_PATH)

def calculate_psi(expected, actual, num_buckets=10):
    """Calcula el Population Stability Index (PSI) entre la distribución baseline y la nueva"""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=buckets)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Reemplazar ceros para evitar divisiones por cero
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)

@app.get("/")
def home():
    return {"message": "API de Segmentación de Clientes Online Retail está activa"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "baseline_data_loaded": baseline_df is not None
    }

@app.post("/predict")
def predict_cluster(data: CustomerData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="El modelo o el escalador no están disponibles")
    
    input_df = pd.DataFrame([data.dict()])
    scaled_data = scaler.transform(input_df)
    cluster = model.predict(scaled_data)
    
    return {
        "cluster": int(cluster[0]),
        "input_summary": data.dict()
    }

@app.post("/drift")
def check_drift(batch_data: list[CustomerData]):
    if baseline_df is None:
        raise HTTPException(status_code=500, detail="No se encontró el dataset baseline de María (customer_features.csv)")
    
    new_data_df = pd.DataFrame([item.dict() for item in batch_data])
    psi_results = {}
    
    for col in CustomerData.__fields__.keys():
        if col in baseline_df.columns:
            psi_val = calculate_psi(baseline_df[col].dropna(), new_data_df[col].dropna())
            
            if psi_val < 0.1:
                status = "Sin cambio significativo"
            elif psi_val < 0.2:
                status = "Deriva moderada"
            else:
                status = "Deriva significativa (Drift detectado)"
                
            psi_results[col] = {
                "psi": round(psi_val, 4),
                "status": status
            }
            
    return {
        "total_records_evaluated": len(new_data_df),
        "drift_analysis": psi_results
    }