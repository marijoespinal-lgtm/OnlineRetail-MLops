import os
import numpy as np
import pandas as pd

def check_retraining_trigger(psi_value: float, metric_value: float, metric_threshold: float = 0.50) -> dict:
    """
    Evalúa si se debe disparar el pipeline de reentrenamiento.
    
    Criterios:
    - PSI > 0.25 (Existe Data Drift significativo)
    - Silhouette / Performance < metric_threshold (Existe degradación de calidad)
    """
    drift_detected = psi_value > 0.25
    degradation_detected = metric_value < metric_threshold

    decision = {
        "psi_observed": psi_value,
        "performance_observed": metric_value,
        "drift_flag": drift_detected,
        "degradation_flag": degradation_detected,
        "action": "NO_ACTION",
        "reason": ""
    }

    # Lógica de decisión combinada
    if drift_detected and degradation_detected:
        decision["action"] = "TRIGGER_RETRAINING"
        decision["reason"] = "ALERT: Data Drift severo y degradación del modelo confirmados. Se dispara reentrenamiento automático."
    
    elif drift_detected and not degradation_detected:
        decision["action"] = "MONITOR_ONLY"
        decision["reason"] = "WARNING: Se detectó Data Drift pero el modelo mantiene su rendimiento. NO se reentrena automáticamente."
    
    elif not drift_detected and degradation_detected:
        decision["action"] = "INVESTIGATE_SYSTEM"
        decision["reason"] = "ALERT: Pérdida de rendimiento sin Data Drift. Posible error en pipeline de datos o cambio en el target."
    
    else:
        decision["action"] = "NO_ACTION"
        decision["reason"] = "OK: Sistema operando dentro de parámetros normales."

    return decision

if __name__ == "__main__":
    print("=" * 70)
    print(" SIMULACION DE DECISION DE REENTRENAMIENTO (PASO R)")
    print("=" * 70)

    # Casos de prueba para evaluar la lógica
    test_cases = [
        {"name": "Caso 1: Operación Normal", "psi": 0.05, "perf": 0.65},
        {"name": "Caso 2: Data Drift sin Degradación (Covariate Shift)", "psi": 0.35, "perf": 0.58},
        {"name": "Caso 3: Data Drift + Degradación Crítica", "psi": 0.42, "perf": 0.38},
    ]

    for test in test_cases:
        res = check_retraining_trigger(test["psi"], test["perf"])
        print(f"\n--- {test['name']} ---")
        print(f"  * PSI: {res['psi_observed']} | Silhouette Score: {res['performance_observed']}")
        print(f"  * Accion: {res['action']}")
        print(f"  * Justificación: {res['reason']}")