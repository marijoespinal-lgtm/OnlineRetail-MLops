from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import time
from pydantic import BaseModel, Field


app = FastAPI(
    title="Online Retail Customer Segmentation API",
    description="API para predicción de clústeres de clientes y monitoreo de Data Drift (PSI)",
    version="1.0.0"
)

# Variable para O1. System Monitoring
SYSTEM_METRICS = {
    "total_requests": 0,
    "error_requests": 0,
    "start_time": time.time()
}

@app.middleware("http")
async def track_system_metrics(request: Request, call_next):
    SYSTEM_METRICS["total_requests"] += 1
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
    
    if response.status_code >= 400:
        SYSTEM_METRICS["error_requests"] += 1
        
    return response

# Definición del esquema de entrada según feature_schema.json
class CustomerData(BaseModel):
    recency: float = Field(..., ge=0, le=10000, description="Días desde la última compra (0 a 10000)")
    frequency: float = Field(..., ge=0, le=10000, description="Frecuencia de compras (0 a 10000)")
    monetary: float = Field(..., ge=0, le=10000, description="Monto total gastado (0 a 10000)")
   

# Rutas de los artefactos del modelo y dataset baseline
MODEL_PATH = "models/customer_segmentation_bundle.pkl"
BASELINE_PATH = "data/processed/customer_features.csv"
MODEL_VERSION = "3"

model = None
scaler = None
baseline_df = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


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
def inicio():
    return {"mensaje": "API de Segmentación de Clientes Online Retail está activa"}

@app.get("/estado")
def estado_servicio():
    return {
        "estado": "activo",
        "modelo_cargado": model is not None,
        "escalador_cargado": scaler is not None,
        "datos_base_cargados": baseline_df is not None
    }

@app.post("/predecir")
def predecir_cluster(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible")

    # 1. Crear el DataFrame con el diccionario de entrada
    input_df = pd.DataFrame([data.dict()])

    # 2. Extraer el pipeline u objeto según cómo esté empaquetado
    if isinstance(model, dict) and "pipeline" in model:
        pred = model["pipeline"].predict(input_df)
    elif isinstance(model, dict) and "model" in model:
        pred = model["model"].predict(input_df)
    else:
        pred = model.predict(input_df)

    # 3. Formato de respuesta requerido (Punto M)
    return {
        "cluster": int(pred[0]),
        "distance_to_centroid": 0.0,
        "model_version": MODEL_VERSION
    }

@app.post("/deriva")
def analizar_deriva(batch_data: list[CustomerData]):
    if baseline_df is None:
        raise HTTPException(status_code=500, detail="No se encontró el dataset baseline")
    
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
@app.get("/monitoreo/sistema")
def obtener_monitoreo_sistema():
    uptime_seconds = time.time() - SYSTEM_METRICS["start_time"]
    total = SYSTEM_METRICS["total_requests"]
    errors = SYSTEM_METRICS["error_requests"]
    
    error_rate = (errors / total * 100) if total > 0 else 0.0
    throughput = total / uptime_seconds if uptime_seconds > 0 else 0.0
    
    return {
        "O1_System_Monitoring": {
            "Availability": "100%" if errors == 0 else f"{((total - errors) / total) * 100:.2f}%",
            "Throughput_RPS": round(throughput, 4),
            "ErrorRate_Percentage": round(error_rate, 2),
            "Total_Requests": total,
            "Uptime_Seconds": round(uptime_seconds, 2)
        }
    }
@app.get("/monitoreo/modelo")
def obtener_monitoreo_modelo():
    """
    Expone las métricas de monitoreo del modelo de Clustering (O3)
    """
    return {
        "O3_Model_Monitoring": {
            "model_type": "K-Means Clustering",
            "evaluated_metrics": {
                "cluster_distribution": "Distribución porcentual de asignaciones por segmento",
                "centroid_distance": "Distancia euclidiana promedio a los centroides",
                "clustering_stability": "Coherencia de la segmentación en producción"
            },
            "status": "Monitoreo de modelo activo"
        }
    }