import pandas as pd
import numpy as np
from src.monitoring.drift_detector import calculate_psi, evaluate_feature_drift

def test_psi_identical_distributions():
    """Verifica que distribuciones similares generen un PSI bajo (< 0.10)."""
    np.random.seed(42)
    ref = pd.Series(np.random.normal(100, 10, 500))
    prod = pd.Series(np.random.normal(100, 10, 500))
    
    psi = calculate_psi(ref, prod)
    assert psi < 0.10, f"El PSI debería ser bajo para distribuciones similares, dio: {psi}"

def test_psi_drift_detection():
    """Verifica que se detecte alerta (PSI >= 0.25) ante un cambio fuerte en los datos."""
    np.random.seed(42)
    ref = pd.Series(np.random.normal(100, 10, 500))
    prod = pd.Series(np.random.normal(200, 50, 500)) # Desplazamiento significativo
    
    psi = calculate_psi(ref, prod)
    assert psi >= 0.25, f"El PSI debería detectar alerta de drift, dio: {psi}"