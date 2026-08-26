"""
feature_sets.py
----------------
- ML & Feature Engineering

Define los conjuntos de variables (feature sets) que se comparan en
el experimento de clustering. Es la UNICA fuente que define "que
variables componen RFM, RFM+ y Behavioral". Tanto train.py como el
notebook de EDA importan estas listas, en vez de escribirlas por su
cuenta en cada lugar (evita la logica duplicada que prohibe la
seccion I de las instrucciones).

Las columnas deben existir (en su version *_scaled) en el CSV que
genera build_features.py.
"""

# RFM clasico: el minimo exigido por la industria, pero insuficiente
# segun las instrucciones del proyecto ("no podran limitar el proyecto
# unicamente a RFM").
RFM = ["recency", "frequency", "monetary"]

# RFM+ : agrega antiguedad del cliente y regularidad/tamano de compra.
RFM_PLUS = RFM + ["tenure", "ipt_mean", "ipt_std", "avg_ticket"]

# Behavioral: el set completo, agregando comportamiento transaccional
# (devoluciones, diversidad, estacionalidad semanal, volumen, precio,
# concentracion de productos).
BEHAVIORAL = RFM_PLUS + [
    "monetary_std", "return_rate", "unique_products",
    "weekend_purchase_pct", "total_items",
    "avg_unit_price", "product_concentration",
]

FEATURE_SETS = {
    "RFM": RFM,
    "RFM_PLUS": RFM_PLUS,
    "BEHAVIORAL": BEHAVIORAL,
}


def get_scaled_columns(feature_set_name: str) -> list:
    """Devuelve los nombres de columnas *_scaled para un feature set dado."""
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(
            f"Feature set '{feature_set_name}' no existe. Opciones: {list(FEATURE_SETS)}"
        )
    return [f"{c}_scaled" for c in FEATURE_SETS[feature_set_name]]
