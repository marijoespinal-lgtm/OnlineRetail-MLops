"""
build_features.py
------------------
- ML & Feature Engineering

Transforma el dataset transaccional limpio (salida de Integrante 1,
en data/clean/online_retail_clean.csv) en un dataset a nivel de
CustomerID con TODAS las variables candidatas para clustering
(RFM + variables adicionales del paso B de las instrucciones).

Este script NO decide que columnas usar para clustering -- eso lo
define src/features/feature_sets.py. Aqui solo se calculan todas
las variables posibles, crudas y escaladas, para que train.py pueda
elegir subconjuntos (RFM / RFM+ / Behavioral) sin recalcular nada.

Uso:
    python src/features/build_features.py \
        --input data/clean/online_retail_clean.csv \
        --output data/processed/customer_features.csv
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

REFERENCE_DATE_BUFFER_DAYS = 1  # snapshot = max(InvoiceDate) + 1 dia

# Todas las columnas candidatas que este script genera.
# feature_sets.py elige subconjuntos de esta lista -- esta es la
# UNICA fuente de verdad de "que variables existen".
ALL_FEATURE_COLUMNS = [
    "recency", "frequency", "monetary",                 # RFM base
    "tenure", "ipt_mean", "ipt_std", "avg_ticket",       # RFM+
    "monetary_std", "return_rate", "unique_products",
    "weekend_purchase_pct", "total_items",
    "avg_unit_price", "product_concentration",           # Behavioral
]


def load_clean_data(path: str) -> pd.DataFrame:
    """Carga el dataset ya limpio por el pipeline de Integrante 1."""
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    required_cols = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "StockCode"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset limpio: {missing}")
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """RFM base: Recency, Frequency, Monetary."""
    grouped = df.groupby("CustomerID")
    rfm = grouped.agg(
        recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        frequency=("InvoiceNo", "nunique"),
        monetary=("TotalPrice", "sum"),
    ).reset_index()
    return rfm


def build_rfm_plus(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Variables RFM+ (paso B): Tenure, regularidad de compra (IPT), ticket promedio."""
    grouped = df.groupby("CustomerID")

    # Tenure: dias desde la primera compra hasta el snapshot
    tenure = grouped.agg(
        first_purchase=("InvoiceDate", "min"),
        n_invoices=("InvoiceNo", "nunique"),
        total_spent=("TotalPrice", "sum"),
    ).reset_index()
    tenure["tenure"] = (reference_date - tenure["first_purchase"]).dt.days
    tenure["avg_ticket"] = tenure["total_spent"] / tenure["n_invoices"]

    # Inter-Purchase Time: dias promedio y variabilidad entre compras consecutivas
    invoice_dates = (
        df.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"].min().reset_index()
        .sort_values(["CustomerID", "InvoiceDate"])
    )
    invoice_dates["days_since_prev"] = (
        invoice_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    )
    ipt = invoice_dates.groupby("CustomerID")["days_since_prev"].agg(
        ipt_mean="mean", ipt_std="std"
    ).reset_index()
    # Clientes con 1 sola compra no tienen intervalo -> se llenan con 0
    ipt["ipt_mean"] = ipt["ipt_mean"].fillna(0)
    ipt["ipt_std"] = ipt["ipt_std"].fillna(0)

    result = tenure[["CustomerID", "tenure", "avg_ticket"]].merge(ipt, on="CustomerID", how="left")
    return result


def build_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Variables comportamentales (ya existentes + nuevas del paso B):
    devoluciones, diversidad de productos, fin de semana, volumen,
    sensibilidad al precio, concentracion de productos."""
    df = df.copy()
    df["is_return"] = df["InvoiceNo"].astype(str).str.startswith("C")
    df["is_weekend"] = df["InvoiceDate"].dt.dayofweek >= 5

    grouped = df.groupby("CustomerID")
    behavioral = grouped.agg(
        n_transactions=("InvoiceNo", "count"),
        n_returns=("is_return", "sum"),
        unique_products=("StockCode", "nunique"),
        weekend_purchases=("is_weekend", "sum"),
        total_items=("Quantity", "sum"),            # Volumen de compra
        avg_unit_price=("UnitPrice", "mean"),        # Sensibilidad al precio
        monetary_std=("TotalPrice", "std"),
    ).reset_index()

    behavioral["return_rate"] = (behavioral["n_returns"] / behavioral["n_transactions"]).fillna(0)
    behavioral["weekend_purchase_pct"] = (
        behavioral["weekend_purchases"] / behavioral["n_transactions"]
    ).fillna(0)
    behavioral["monetary_std"] = behavioral["monetary_std"].fillna(0)

    # Concentracion de productos: que % de sus items totales corresponde
    # a su producto mas comprado (cercano a 1 = compra casi siempre lo mismo,
    # cercano a 0 = compras muy dispersas entre productos distintos)
    top_product_qty = (
        df.groupby(["CustomerID", "StockCode"])["Quantity"].sum()
        .reset_index()
        .groupby("CustomerID")["Quantity"].max()
        .reset_index()
        .rename(columns={"Quantity": "top_product_qty"})
    )
    behavioral = behavioral.merge(top_product_qty, on="CustomerID", how="left")
    behavioral["product_concentration"] = (
        behavioral["top_product_qty"] / behavioral["total_items"]
    ).fillna(0)

    return behavioral[[
        "CustomerID", "monetary_std", "return_rate", "unique_products",
        "weekend_purchase_pct", "total_items", "avg_unit_price",
        "product_concentration",
    ]]


def scale_features(df: pd.DataFrame, feature_cols: list, scaler_path: str = None):
    """Escala TODAS las columnas candidatas con un unico StandardScaler
    (el escalado es independiente por columna, asi que un solo scaler
    sirve sin importar que subconjunto se use despues para clustering)."""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in feature_cols])
    result = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info("Scaler guardado en %s", scaler_path)

    return result, scaler


# Variables con cola larga (unos pocos clientes con valores extremos que
# distorsionan el StandardScaler y "estiran" las distancias en el clustering,
# forzando micro-clusters de puros outliers). Se capan a los percentiles
# 1 y 99 -- no se eliminan filas, solo se recorta el valor extremo.
OUTLIER_PRONE_COLUMNS = [
    "monetary", "avg_ticket", "ipt_mean", "ipt_std",
    "monetary_std", "total_items", "avg_unit_price",
]


def cap_outliers(df: pd.DataFrame, columns: list, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """Winsoriza (recorta) valores extremos a los percentiles dados,
    columna por columna. No elimina clientes, solo limita que tan
    extremo puede ser un valor individual."""
    df = df.copy()
    for col in columns:
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_capped > 0:
            logger.info(
                "Capando outliers en '%s': %d clientes fuera de [%.2f, %.2f]",
                col, n_capped, lower, upper,
            )
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def build_customer_features(df: pd.DataFrame, scaler_path: str = None):
    """Pipeline completo: raw transaccional -> TODAS las variables candidatas,
    crudas y escaladas."""
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=REFERENCE_DATE_BUFFER_DAYS)

    rfm = build_rfm(df, reference_date)
    rfm_plus = build_rfm_plus(df, reference_date)
    behavioral = build_behavioral(df)

    features = rfm.merge(rfm_plus, on="CustomerID", how="left").merge(
        behavioral, on="CustomerID", how="left"
    )

    features[ALL_FEATURE_COLUMNS] = features[ALL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    features = features.dropna(subset=ALL_FEATURE_COLUMNS)

    features = cap_outliers(features, OUTLIER_PRONE_COLUMNS)

    features_scaled, scaler = scale_features(features, ALL_FEATURE_COLUMNS, scaler_path)
    return features_scaled, scaler, ALL_FEATURE_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Feature engineering para clustering de clientes")
    parser.add_argument("--input", default="data/clean/online_retail_clean.csv")
    parser.add_argument("--output", default="data/processed/customer_features.csv")
    parser.add_argument("--scaler-output", default="models/scaler.pkl")
    args = parser.parse_args()

    logger.info("Cargando datos limpios desde %s", args.input)
    df = load_clean_data(args.input)

    logger.info("Construyendo todas las variables candidatas...")
    features, _, feature_cols = build_customer_features(df, scaler_path=args.scaler_output)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    logger.info(
        "Features guardadas en %s (%d clientes, %d variables candidatas)",
        args.output, len(features), len(feature_cols),
    )


if __name__ == "__main__":
    main()
