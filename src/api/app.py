from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Online Retail Customer Segmentation API",
    description="API para predicción de clústeres de clientes usando datos RFM extendidos",
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

# Carga del modelo y escalador al iniciar la aplicación
MODEL_PATH = "models/customer_segmentation_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

@app.get("/")
def home():
    return {"message": "API de Segmentación de Clientes Online Retail está activa"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict")
def predict_cluster(data: CustomerData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="El modelo o el escalador no están disponibles")
    
    # Convertir datos de entrada a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Aplicar transformaciones
    scaled_data = scaler.transform(input_df)
    
    # Inferencia
    cluster = model.predict(scaled_data)
    
    return {
        "cluster": int(cluster[0]),
        "input_summary": data.dict()
    }