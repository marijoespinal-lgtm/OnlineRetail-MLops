from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Online Retail Customer Segmentation API",
    description="API de inferencia bajo principios MLOps",
    version="1.0.0"
)

class CustomerFeatures(BaseModel):
    customer_id: str
    recency: float
    frequency: int
    monetary: float
    weekend_ratio: float = 0.0
    product_diversity: int = 1

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Online Retail ML API"}

@app.post("/predict")
def predict_cluster(data: CustomerFeatures):
    # Lógica temporal hasta integrar el modelo guardado por MLflow
    return {
        "customer_id": data.customer_id,
        "assigned_cluster": 1,
        "distance_to_centroid": 0.321,
        "model_version": "v1.0-baseline"
    }