## Este script valida el contrato de datos del modelo
def test_customer_features_schema():
    """Valida la integridad de las variables calculadas por cliente."""
    sample_customer = {
        "recency": 15.0,
        "frequency": 2,
        "monetary": 120.0
    }
    
    # Assertions para Data Quality antes de inferencia
    assert sample_customer["recency"] >= 0, "La recencia no puede ser negativa"
    assert sample_customer["frequency"] > 0, "La frecuencia debe ser mayor a 0"
    assert sample_customer["monetary"] >= 0, "El valor monetario debe ser no negativo"