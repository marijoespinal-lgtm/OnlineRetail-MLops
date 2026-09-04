"""
train.py
--------
Integrante 2 - ML & Experiment Tracking

Compara clustering usando 3 conjuntos de variables (RFM / RFM+ /
Behavioral, definidos en src/features/feature_sets.py) x 3 algoritmos
(K-Means, Gaussian Mixture, Agglomerative Clustering), registrando
todo en MLflow.

Ciclo completo implementado:

Experiment -> Candidate -> Validation -> Champion -> Production

Uso:
python src/models/train.py --data data/processed/customer_features.csv
mlflow ui # para inspeccionar los runs
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg") # sin ventanas graficas -- evita los errores de tkinter en Windows
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.features.feature_sets import FEATURE_SETS, get_scaled_columns # noqa: E402
from src.models.validation import validate_candidate # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# IMPORTANTE: apunta explicitamente al servidor de MLflow (mlflow server
# --host 127.0.0.1 --port 5000). Sin esto, train.py escribe a una carpeta
# local ./mlruns que el servidor NUNCA lee, y los runs nuevos no aparecen
# en la interfaz web aunque el script diga que corrio bien.
# El servidor debe estar corriendo ANTES de ejecutar este script.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

RANDOM_SEED = 42
EXPERIMENT_NAME = "customer_segmentation_v2"
DATA_VERSION = "v2_rfm_plus_behavioral" # subir version cuando cambien las features
MODEL_REGISTRY_NAME = "customer_segmentation_model"


def evaluate_clustering(X, labels):
"""Silhouette y Davies-Bouldin. Requieren >=2 clusters validos."""
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
if n_clusters < 2:
return None, None
mask = labels != -1
if mask.sum() < 2:
return None, None
sil = silhouette_score(X[mask], labels[mask])
db = davies_bouldin_score(X[mask], labels[mask])
return sil, db


def plot_cluster_distribution(labels, run_name):
fig, ax = plt.subplots(figsize=(6, 4))
pd.Series(labels).value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title(f"Distribucion de clusters - {run_name}")
ax.set_xlabel("Cluster")
ax.set_ylabel("Cantidad de clientes")
path = f"{tempfile.gettempdir()}/cluster_dist_{run_name}.png"
fig.savefig(path, bbox_inches="tight")
plt.close(fig)
return path


def build_cluster_profile(df, labels, feature_set_cols_raw):
"""Perfil de negocio: promedio de cada variable (SIN escalar) por cluster.
Esto es lo que responde 'que significa cada cluster', no solo metricas."""
profile_df = df.copy()
profile_df["cluster"] = labels
profile = profile_df.groupby("cluster")[feature_set_cols_raw].mean().round(2)
profile["n_clientes"] = profile_df.groupby("cluster").size()
return profile


def run_experiment(name, model, X, df_raw, feature_set_name, feature_set_cols_raw, params):
with mlflow.start_run(run_name=name):
mlflow.log_param("algorithm", name.split("__")[0])
mlflow.log_param("feature_set", feature_set_name)
mlflow.log_param("random_seed", RANDOM_SEED)
mlflow.log_param("data_version", DATA_VERSION)
for k, v in params.items():
mlflow.log_param(k, v)

labels = model.fit_predict(X)

sil, db = evaluate_clustering(X, labels)
if sil is not None:
mlflow.log_metric("silhouette_score", sil)
mlflow.log_metric("davies_bouldin_score", db)
else:
logger.warning("%s: no se pudo calcular metricas (menos de 2 clusters validos)", name)

n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
mlflow.log_metric("n_clusters_found", n_clusters_found)

# Artifact: grafico de distribucion de clusters
plot_path = plot_cluster_distribution(labels, name)
mlflow.log_artifact(plot_path, artifact_path="plots")

# Artifact: perfil de negocio de cada cluster (cluster analysis)
profile = build_cluster_profile(df_raw, labels, feature_set_cols_raw)
profile_path = f"{tempfile.gettempdir()}/cluster_profile_{name}.csv"
profile.to_csv(profile_path)
mlflow.log_artifact(profile_path, artifact_path="cluster_analysis")

# Artifact: modelo
mlflow.sklearn.log_model(model, artifact_path="model")

logger.info(
"%s -> silhouette=%s, davies_bouldin=%s, clusters=%s",
name, sil, db, n_clusters_found,
)
return mlflow.active_run().info.run_id, sil, model, labels


def main():
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/processed/customer_features.csv")
args = parser.parse_args()

mlflow.set_experiment(EXPERIMENT_NAME)

df = pd.read_csv(args.data)

results = {} # run_id -> silhouette
run_info = {} # run_id -> (model, labels, feature_set_name, feature_set_cols_raw, algorithm)

# --- EXPERIMENT: probar cada feature set x cada algoritmo ---
for feature_set_name, raw_cols in FEATURE_SETS.items():
scaled_cols = get_scaled_columns(feature_set_name)
X = df[scaled_cols].values

# 1. K-Means
for k in [3, 4, 5]:
model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
name = f"kmeans__{feature_set_name}__k{k}"
run_id, sil, fitted, labels = run_experiment(
name, model, X, df, feature_set_name, raw_cols, {"n_clusters": k}
)
results[run_id] = sil if sil is not None else -1
run_info[run_id] = (fitted, labels, feature_set_name, raw_cols, "kmeans")

# 2. Gaussian Mixture
for k in [3, 4, 5]:
model = GaussianMixture(n_components=k, random_state=RANDOM_SEED)
name = f"gmm__{feature_set_name}__k{k}"
run_id, sil, fitted, labels = run_experiment(
name, model, X, df, feature_set_name, raw_cols, {"n_components": k}
)
results[run_id] = sil if sil is not None else -1
run_info[run_id] = (fitted, labels, feature_set_name, raw_cols, "gmm")

# 3. Agglomerative Clustering
# NOTA: Agglomerative no tiene metodo .predict() -- no puede
# asignar cluster a un cliente NUEVO en produccion (solo agrupa
# los datos que ya tenia al momento de entrenar). Se sigue
# registrando en MLflow para comparar metricas, pero NUNCA se
# promueve como Champion/Production, sin importar su silhouette.
for k in [3, 4, 5]:
model = AgglomerativeClustering(n_clusters=k)
name = f"agglomerative__{feature_set_name}__k{k}"
run_id, sil, fitted, labels = run_experiment(
name, model, X, df, feature_set_name, raw_cols, {"n_clusters": k}
)
results[run_id] = sil if sil is not None else -1
run_info[run_id] = (fitted, labels, feature_set_name, raw_cols, "agglomerative")

# --- CANDIDATE: recorrer runs de mejor a peor silhouette, saltando
# los que no pueden promoverse (Agglomerative), hasta encontrar uno
# que ademas apruebe la Validacion ---
promotable_run_ids = [
run_id for run_id, info in run_info.items() if info[4] != "agglomerative"
]
sorted_candidates = sorted(promotable_run_ids, key=lambda r: results[r], reverse=True)

champion_run_id = None
for run_id in sorted_candidates:
sil = results[run_id]
model, labels, feature_set_name, raw_cols, algorithm = run_info[run_id]
logger.info(
"CANDIDATE: run %s (algorithm=%s, feature_set=%s, silhouette=%.4f)",
run_id, algorithm, feature_set_name, sil,
)
passed, reasons = validate_candidate(labels, sil if sil != -1 else None)
if passed:
champion_run_id = run_id
break
logger.warning(
"Candidato run %s RECHAZADO en validacion, probando el siguiente. Motivos: %s",
run_id, reasons,
)

if champion_run_id is None:
logger.warning(
"NINGUN candidato paso la validacion (de %d posibles, excluyendo Agglomerative). "
"No se promovera nada a Production. Revisar criterios de Validation o el "
"feature engineering.", len(sorted_candidates),
)
return

best_run_id = champion_run_id
best_sil = results[best_run_id]
best_model, best_labels, best_feature_set, best_raw_cols, best_algorithm = run_info[best_run_id]

logger.info(
"CHAMPION confirmado: run %s (algorithm=%s, feature_set=%s, silhouette=%.4f) "
"-> promoviendo a Production",
best_run_id, best_algorithm, best_feature_set, best_sil,
)

# --- MODEL REGISTRY: registrar y promover ---
model_uri = f"runs:/{best_run_id}/model"
registered = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
name=MODEL_REGISTRY_NAME,
version=registered.version,
stage="Production",
archive_existing_versions=True,
)
logger.info("Modelo version %s registrado y promovido a Production", registered.version)

# --- Empaquetar TODO junto: scaler + feature_set + modelo + labels ---
# (un solo artefacto, en vez de repartir scaler/modelo por separado,
# como pide el diagrama de "Customer Segmentation Model")
export_dir = Path("models")
export_dir.mkdir(exist_ok=True)

scaler = joblib.load(export_dir / "scaler.pkl")
bundle = {
"model": best_model,
"scaler": scaler,
"algorithm": best_algorithm,
"feature_set_name": best_feature_set,
"feature_columns_raw": best_raw_cols,
"feature_columns_scaled": get_scaled_columns(best_feature_set),
"model_version": str(registered.version),
"silhouette_score": best_sil,
}
joblib.dump(bundle, export_dir / "customer_segmentation_bundle.pkl")
logger.info("Bundle completo exportado a %s", export_dir / "customer_segmentation_bundle.pkl")

# Perfil de negocio del modelo ganador (para Integrante 3 / README)
profile = build_cluster_profile(df, best_labels, best_raw_cols)
profile.to_csv(export_dir / "champion_cluster_profile.csv")

# Contrato de columnas para la API
contract = {
"model_version": str(registered.version),
"feature_set": best_feature_set,
"required_input_columns": best_raw_cols,
"note": (
"El input debe traer estas columnas SIN escalar. La API debe "
"cargar customer_segmentation_bundle.pkl y usar bundle['scaler'] "
"y bundle['feature_columns_scaled'] para transformar antes de "
"llamar a bundle['model'].predict()."
),
}
with open(export_dir / "feature_schema.json", "w") as f:
json.dump(contract, f, indent=2)
logger.info("Contrato de columnas actualizado en %s", export_dir / "feature_schema.json")


if __name__ == "__main__":
main()