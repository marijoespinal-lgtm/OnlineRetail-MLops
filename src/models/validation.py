"""
validation.py
--------------
- Model Registry

Implementa la etapa "Validation" del ciclo:

    Experiment -> Candidate -> Validation -> Champion -> Production

Aqui se definen criterios EXPLICITOS que el candidato debe cumplir antes de
convertirse en "campeon" y pasar a Production, tal como exige la
seccion K de las instrucciones ("la seleccion debera basarse en
criterios explicitos").
"""

import logging

logger = logging.getLogger(__name__)

# Umbrales justificados (no son leyes universales -- ver README):
# - Silhouette minimo de 0.4: por debajo de eso, sklearn considera que
#   los clusters ya no tienen una separacion razonable.
# - Ningun cluster con menos del 3% de los clientes: un cluster
#   minusculo no es un segmento accionable para negocio (no vale la
#   pena una campana de marketing para 10 clientes).
# - Entre 2 y 8 clusters: menos de 2 no es clustering, y mas de 8
#   deja de ser interpretable/accionable para una estrategia comercial.
MIN_SILHOUETTE = 0.4
MIN_CLUSTER_SIZE_PCT = 0.03
MIN_CLUSTERS = 2
MAX_CLUSTERS = 8


def validate_candidate(labels, silhouette_score_value: float) -> tuple[bool, list[str]]:
    """Valida un modelo candidato contra los criterios explicitos.

    Devuelve (paso_validacion: bool, razones: list[str]) -- si paso_validacion
    es False, train.py NO debe promover el modelo a Production
    automaticamente; debe registrar el motivo y dejarlo para revision.
    """
    reasons = []

    if silhouette_score_value is None:
        reasons.append("Silhouette no calculable (menos de 2 clusters validos)")
    elif silhouette_score_value < MIN_SILHOUETTE:
        reasons.append(
            f"Silhouette {silhouette_score_value:.4f} por debajo del minimo {MIN_SILHOUETTE}"
        )

    import numpy as np
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    n_clusters = len(unique_labels)
    total = counts.sum()

    if n_clusters < MIN_CLUSTERS:
        reasons.append(f"Solo {n_clusters} cluster(s) encontrados (minimo {MIN_CLUSTERS})")
    if n_clusters > MAX_CLUSTERS:
        reasons.append(f"{n_clusters} clusters encontrados (maximo {MAX_CLUSTERS} por interpretabilidad)")

    if total > 0:
        min_pct = counts.min() / total
        if min_pct < MIN_CLUSTER_SIZE_PCT:
            reasons.append(
                f"El cluster mas pequeno tiene {min_pct:.1%} de los clientes "
                f"(minimo {MIN_CLUSTER_SIZE_PCT:.0%})"
            )

    passed = len(reasons) == 0
    if passed:
        logger.info("Candidato APROBADO en validacion (Champion)")
    else:
        logger.warning("Candidato RECHAZADO en validacion. Motivos: %s", reasons)

    return passed, reasons
