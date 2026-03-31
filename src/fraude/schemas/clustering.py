# src/fraude/clustering_schema.py
"""
Modelos Pydantic para el módulo de Clustering de Fraude.

Define contratos de entrada/salida del endpoint /fraude/clustering/compute.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────
# REQUEST
# ──────────────────────────────────────────────────────────────────

class ClusteringRequest(BaseModel):
    """
    Parámetros opcionales para el análisis de clustering.
    Todos tienen defaults seguros para producción.
    """
    n_clusters: int = Field(
        default=3,
        ge=2,
        le=8,
        description="Número de clusters a generar (K-Means). Entre 2 y 8.",
    )
    min_samples: int = Field(
        default=30,
        ge=10,
        description="Mínimo de muestras de fraude requeridas para ejecutar el análisis.",
    )
    lookback_days: Optional[int] = Field(
        default=None,
        description="Si se especifica, limita el análisis a los últimos N días. "
                    "None = usa toda la historia disponible.",
    )


# ──────────────────────────────────────────────────────────────────
# RESPONSE
# ──────────────────────────────────────────────────────────────────

class ClusterProfile(BaseModel):
    """
    Perfil de un cluster de defraudadores.
    Cada campo representa el valor promedio (centroid) del cluster.
    """
    cluster_id:      int
    label:           str           # Label generado automáticamente, e.g. "Fraude nocturno de alto monto"
    fraud_count:     int           # Transacciones en este cluster
    pct_of_total:    float         # Porcentaje respecto al total de fraudes analizados

    # Centroids numéricos
    avg_amount:      float
    avg_hour:        float
    avg_age:         float
    avg_distance_km: float
    avg_city_pop:    float

    # Dominantes categóricos (moda del cluster)
    top_category:    Optional[str] = None
    top_state:       Optional[str] = None


class ClusteringResponse(BaseModel):
    """Respuesta completa del análisis de clustering."""
    profiles:         List[ClusterProfile]
    total_frauds_analyzed: int
    n_clusters_used:  int
    run_date:         str    # ISO timestamp
    message:          str    # Mensaje informativo para logs/UI