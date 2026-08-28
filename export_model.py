import mlflow
import joblib

# Cargar el modelo ganador desde MLflow (Production)
model_uri = "models:/customer_segmentation_model/Production"
model = mlflow.sklearn.load_model(model_uri)

# Guardar el modelo en la carpeta que usa tu API
joblib.dump(model, "models/customer_segmentation_model.pkl")

