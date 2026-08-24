import numpy as np
import pandas as pd

def calculate_psi(reference: pd.Series, production: pd.Series, num_buckets: int = 10) -> float:
    """
    Calcula el Population Stability Index (PSI) entre la distribución de referencia
    y la distribución de producción.
    
    Interpretación:
    - PSI < 0.10: Sin cambio significativo (OK)
    - 0.10 <= PSI < 0.25: Cambio moderado / Alerta (WARNING)
    - PSI >= 0.25: Cambio crítico (ALERT)
    """
    # Eliminar nulos para el cálculo
    ref_clean = reference.dropna()
    prod_clean = production.dropna()
    
    # Definir los cuantiles basados en el dataset de referencia
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(ref_clean, percentiles)
    
    # Ajustar bordes para evitar errores de límites
    buckets[0] = buckets[0] - 1e-5
    buckets[-1] = buckets[-1] + 1e-5

    # Calcular proporciones en cada bucket
    ref_counts, _ = np.histogram(ref_clean, bins=buckets)
    prod_counts, _ = np.histogram(prod_clean, bins=buckets)

    ref_pct = ref_counts / len(ref_clean)
    prod_pct = prod_counts / len(prod_clean)

    # Reemplazar ceros con un valor minúsculo para evitar división por cero o log(0)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-4, prod_pct)

    # Fórmula del PSI
    psi_value = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi_value)


def evaluate_feature_drift(ref_df: pd.DataFrame, prod_df: pd.DataFrame, threshold_warning: float = 0.10, threshold_alert: float = 0.25):
    """
    Evalúa el drift para todas las variables numéricas entre Reference y Production.
    """
    results = {}
    numeric_cols = ref_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col in prod_df.columns:
            psi = calculate_psi(ref_df[col], prod_df[col])
            
            if psi >= threshold_alert:
                status = "ALERT"
            elif psi >= threshold_warning:
                status = "WARNING"
            else:
                status = "OK"

            results[col] = {
                "psi": round(psi, 4),
                "status": status
            }
            
    return results


if __name__ == "__main__":
    # Simulación de prueba local
    np.random.seed(42)
    df_ref = pd.DataFrame({"monetary": np.random.normal(100, 15, 1000)})
    df_prod_ok = pd.DataFrame({"monetary": np.random.normal(101, 15, 1000)})
    df_prod_drift = pd.DataFrame({"monetary": np.random.normal(180, 40, 1000)})

    print("--- Test PSI Sin Drift ---")
    print(evaluate_feature_drift(df_ref, df_prod_ok))

    print("\n--- Test PSI Con Drift Detectado ---")
    print(evaluate_feature_drift(df_ref, df_prod_drift))