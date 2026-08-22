"""
train.py
--------
Integrante 2 - ML & Experiment Tracking

Entrena y compara al menos 3 algoritmos de clustering sobre las
features de clientes, registrando todo en MLflow (parámetros, métricas,
gráficos y artefactos), y promueve el mejor modelo vía Model Registry.

Uso:
    python src/models/train.py --data data/processed/customer_features.csv
    mlflow ui   # para inspeccionar los runs
"""

import argparse
import logging

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
EXPERIMENT_NAME = "customer_segmentation"
DATA_VERSION = "v1"  # actualizar si cambia el dataset/features


def get_feature_matrix(df: pd.DataFrame, feature_cols):
    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    return df[scaled_cols].values


def evaluate_clustering(X, labels):
    """Silhouette y Davies-Bouldin. Ambos requieren >=2 clusters válidos
    (DBSCAN puede producir solo ruido o un cluster; se maneja ese caso)."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return None, None
    mask = labels != -1  # excluir ruido de DBSCAN al evaluar
    if mask.sum() < 2:
        return None, None
    sil = silhouette_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    return sil, db


def plot_cluster_distribution(labels, run_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(labels).value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title(f"Distribución de clusters - {run_name}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cantidad de clientes")
    path = f"/tmp/cluster_dist_{run_name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_experiment(name, model, X, feature_set, params):
    with mlflow.start_run(run_name=name):
        mlflow.log_param("algorithm", name)
        mlflow.log_param("feature_set", feature_set)
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
            logger.warning("%s: no se pudo calcular métricas (menos de 2 clusters válidos)", name)

        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        mlflow.log_metric("n_clusters_found", n_clusters_found)

        plot_path = plot_cluster_distribution(labels, name)
        mlflow.log_artifact(plot_path, artifact_path="plots")

        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("%s -> silhouette=%s, davies_bouldin=%s, clusters=%s", name, sil, db, n_clusters_found)
        return mlflow.active_run().info.run_id, sil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/customer_features.csv")
    args = parser.parse_args()

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(args.data)
    feature_cols = ["recency", "frequency", "monetary", "monetary_std",
                     "return_rate", "unique_products", "weekend_purchase_pct"]
    X = get_feature_matrix(df, feature_cols)
    feature_set = ",".join(feature_cols)

    results = {}

    # 1. K-Means
    for k in [3, 4, 5]:
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        run_id, sil = run_experiment(f"kmeans_k{k}", model, X, feature_set, {"n_clusters": k})
        results[run_id] = sil or -1

    # 2. DBSCAN
    for eps in [0.5, 0.8]:
        model = DBSCAN(eps=eps, min_samples=5)
        run_id, sil = run_experiment(f"dbscan_eps{eps}", model, X, feature_set, {"eps": eps, "min_samples": 5})
        results[run_id] = sil or -1

    # 3. Agglomerative Clustering
    for k in [3, 4, 5]:
        model = AgglomerativeClustering(n_clusters=k)
        run_id, sil = run_experiment(f"agglomerative_k{k}", model, X, feature_set, {"n_clusters": k})
        results[run_id] = sil or -1

    best_run_id = max(results, key=results.get)
    logger.info("Mejor run: %s (silhouette=%.4f)", best_run_id, results[best_run_id])

    # --- Model Registry: registrar y promover el mejor modelo ---
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name="customer_segmentation_model")

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="customer_segmentation_model",
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        "Modelo version %s registrado y promovido a Production", registered.version
    )


if __name__ == "__main__":
    main()
