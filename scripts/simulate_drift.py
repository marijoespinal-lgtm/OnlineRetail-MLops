import os
import numpy as np
import pandas as pd

BASELINE_PATH = "data/processed/customer_features.csv"

if os.path.exists(BASELINE_PATH):
    df_ref = pd.read_csv(BASELINE_PATH)
else:
    np.random.seed(42)
    df_ref = pd.DataFrame({
        "recency": np.random.exponential(scale=30, size=1000),
        "frequency": np.random.poisson(lam=3, size=1000),
        "monetary": np.random.normal(loc=200, scale=50, size=1000)
    })

def calculate_psi(expected, actual, num_buckets=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(expected, percentiles)
    buckets[0] = -np.inf
    buckets[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=buckets)
    actual_counts, _ = np.histogram(actual, bins=buckets)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

def evaluate_status(psi_val):
    if psi_val < 0.10:
        return f"PSI = {psi_val:.4f} -> [OK]"
    elif psi_val <= 0.25:
        return f"PSI = {psi_val:.4f} -> [WARNING]"
    else:
        return f"PSI = {psi_val:.4f} -> [ALERT (Drift Detectado)]"

np.random.seed(123)
n_samples = len(df_ref)

batch_1 = pd.DataFrame({
    "recency": df_ref["recency"] + np.random.normal(0, 2, n_samples),
    "frequency": df_ref["frequency"],
    "monetary": df_ref["monetary"] + np.random.normal(0, 5, n_samples)
})

batch_2 = pd.DataFrame({
    "recency": df_ref["recency"] * 1.25,
    "frequency": df_ref["frequency"] * 1.15,
    "monetary": df_ref["monetary"] * 1.20
})

batch_3 = pd.DataFrame({
    "recency": df_ref["recency"] * 2.50,
    "frequency": df_ref["frequency"] * 0.40,
    "monetary": df_ref["monetary"] * 4.00
})

batches = {
    "PRODUCTION BATCH 1 (Comportamiento Normal)": batch_1,
    "PRODUCTION BATCH 2 (Cambio Leve de Patrón)": batch_2,
    "PRODUCTION BATCH 3 (Shift Severo en P(X))": batch_3
}

print("=" * 65)
print(" SIMULACION DE PRODUCCION Y DETECCION DE DATA DRIFT (P(X))")
print("=" * 65)

for name, b_df in batches.items():
    print(f"\n--- {name} ---")
    for col in ["recency", "frequency", "monetary"]:
        if col in df_ref.columns:
            psi = calculate_psi(df_ref[col].dropna(), b_df[col].dropna())
            status = evaluate_status(psi)
            print(f"  * {col}: {status}")