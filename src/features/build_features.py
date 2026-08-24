"""
build_features.py
------------------
Integrante 2 - ML & Feature Engineering

Transforma el dataset transaccional limpio (salida de Integrante 1,
normalmente en data/processed/clean_data.csv) en un dataset a nivel
de CustomerID listo para clustering.

Uso:
    python src/features/build_features.py \
        --input data/processed/clean_data.csv \
        --output data/processed/customer_features.csv

Este módulo expone funciones reutilizables (build_rfm, build_behavioral,
scale_features) para que la MISMA lógica se use en el notebook de EDA,
en el entrenamiento (train.py) y en la API de inferencia — evitando
tener una lógica de features "de notebook" y otra "de producción".
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_DATE_BUFFER_DAYS = 1  # snapshot = max(InvoiceDate) + 1 día


def load_clean_data(path: str) -> pd.DataFrame:
    """Carga el dataset ya limpio por el pipeline de Integrante 1."""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    required_cols = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "StockCode"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset limpio: {missing}")
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """RFM ampliado: Recency, Frequency, Monetary + varianza del gasto."""
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=REFERENCE_DATE_BUFFER_DAYS)

    grouped = df.groupby("CustomerID")
    rfm = grouped.agg(
        recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("TotalPrice", "sum"),
        monetary_std=("TotalPrice", "std"),
    ).reset_index()

    rfm["monetary_std"] = rfm["monetary_std"].fillna(0)
    return rfm


def build_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Variables comportamentales: tasa de devoluciones, diversidad de productos,
    % de compras en fin de semana."""
    df = df.copy()
    df["is_return"] = df["InvoiceNo"].astype(str).str.startswith("C")
    df["is_weekend"] = df["InvoiceDate"].dt.dayofweek >= 5

    grouped = df.groupby("CustomerID")
    behavioral = grouped.agg(
        n_transactions=("InvoiceNo", "count"),
        n_returns=("is_return", "sum"),
        unique_products=("StockCode", "nunique"),
        weekend_purchases=("is_weekend", "sum"),
    ).reset_index()

    behavioral["return_rate"] = (behavioral["n_returns"] / behavioral["n_transactions"]).fillna(0)
    behavioral["weekend_purchase_pct"] = (
        behavioral["weekend_purchases"] / behavioral["n_transactions"]
    ).fillna(0)

    return behavioral[
        ["CustomerID", "return_rate", "unique_products", "weekend_purchase_pct"]
    ]


def merge_features(rfm: pd.DataFrame, behavioral: pd.DataFrame) -> pd.DataFrame:
    return rfm.merge(behavioral, on="CustomerID", how="left")


def scale_features(df: pd.DataFrame, feature_cols: list, scaler_path: str = None):
    """Aplica StandardScaler y opcionalmente guarda el scaler (joblib) para
    reutilizarlo en train.py y en la API de inferencia — mismo escalado
    en entrenamiento y producción, evitando train/serving skew."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in feature_cols])
    result = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler guardado en %s", scaler_path)

    return result, scaler


def build_customer_features(df: pd.DataFrame, scaler_path: str = None):
    """Pipeline completo: raw transaccional -> features por cliente escaladas."""
    rfm = build_rfm(df)
    behavioral = build_behavioral(df)
    features = merge_features(rfm, behavioral)

    feature_cols = [
        "recency", "frequency", "monetary", "monetary_std",
        "return_rate", "unique_products", "weekend_purchase_pct",
    ]
    features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan)
    features = features.dropna(subset=feature_cols)

    features_scaled, scaler = scale_features(features, feature_cols, scaler_path)
    return features_scaled, scaler, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Feature engineering para clustering de clientes")
    parser.add_argument("--input", default="data/clean/online_retail_clean.csv")
    parser.add_argument("--output", default="data/processed/customer_features.csv")
    parser.add_argument("--scaler-output", default="models/scaler.pkl")
    args = parser.parse_args()

    logger.info("Cargando datos limpios desde %s", args.input)
    df = load_clean_data(args.input)

    logger.info("Construyendo features por cliente...")
    features, _, feature_cols = build_customer_features(df, scaler_path=args.scaler_output)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    logger.info("Features guardadas en %s (%d clientes, %d features)", args.output, len(features), len(feature_cols))


if __name__ == "__main__":
    main()
