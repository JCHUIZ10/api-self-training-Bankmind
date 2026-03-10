# src/fraude/clustering_service.py
"""
Servicio de Clustering de Fraude (K-Means).

Responsabilidad única:
  - Extraer transacciones ALTO RIESGO de la BD.
  - Normalizar features y aplicar K-Means.
  - Interpretar centroids y generar labels legibles.
  - Devolver ClusteringResponse listo para serializar.

Flujo de uso:
  Java Scheduler → POST /fraude/clustering/compute → clustering_service.py
  clustering_service → BD (fraud_predictions + operational_transactions)
  clustering_service → K-Means → ClusteringResponse
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from fraude.clustering_schema import ClusterProfile, ClusteringRequest, ClusteringResponse
from fraude.db_config import get_db_connection

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────

FEATURES = ["amt", "hour", "age", "distance_km", "city_pop"]

# Umbral para clasificar monto: percentil 75 del promedio histórico.
# Se recalcula dinámicamente desde los datos, no hardcodeado.
_HIGH_AMOUNT_QUANTILE = 0.75
_NIGHT_HOURS = (20, 6)   # >= 20 o < 6 = nocturno

# ──────────────────────────────────────────────────────────────────
# SQL DE EXTRACCIÓN
# ──────────────────────────────────────────────────────────────────

def _build_sql(lookback_days: Optional[int]) -> tuple:
    """
    Devuelve (sql, params). Extrae solo las transacciones ALTO RIESGO
    con todas las features numéricas necesarias para K-Means.
    """
    date_filter = ""
    params: list = []

    if lookback_days is not None:
        cutoff = datetime.now() - timedelta(days=lookback_days)
        date_filter = "AND fp.prediction_date >= %s"
        params.append(cutoff)

    sql = f"""
        SELECT
            ot.amt,
            EXTRACT(HOUR FROM ot.trans_date_time)                              AS hour,
            DATE_PART('year', AGE(ot.trans_date_time::date, cu.dob))           AS age,
            6371.0 * 2 * ASIN(SQRT(
                POWER(SIN(RADIANS((ot.merch_lat    - l.customer_lat ) / 2)), 2)
              + COS(RADIANS(l.customer_lat))
              * COS(RADIANS(ot.merch_lat))
              * POWER(SIN(RADIANS((ot.merch_long   - l.customer_long) / 2)), 2)
            ))                                                                  AS distance_km,
            l.city_pop,
            cat.category_name                                                   AS category,
            l.state
        FROM fraud_predictions fp
        JOIN operational_transactions ot  ON fp.id_transaction  = ot.id_transaction
        JOIN credit_cards           cc    ON ot.cc_num           = cc.cc_num
        JOIN customer               cu    ON cc.id_customer      = cu.id_customer
        JOIN localization            l    ON cu.id_localization  = l.id_localization
        LEFT JOIN categories        cat   ON ot.id_category      = cat.id_category
        WHERE fp.veredicto = 'ALTO RIESGO'
          AND cu.dob IS NOT NULL
          AND ot.merch_lat IS NOT NULL
          {date_filter}
    """
    return sql, params


# ──────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE DATOS
# ──────────────────────────────────────────────────────────────────

def _load_fraud_data(lookback_days: Optional[int]) -> pd.DataFrame:
    """
    Carga el DataFrame de transacciones ALTO RIESGO desde PostgreSQL.

    Usa cursor.execute + pd.DataFrame, igual que data_extraction.py,
    ya que pd.read_sql_query no es compatible con conexiones psycopg2 directas.
    """
    sql, params = _build_sql(lookback_days)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params if params else None)
        columns = [desc[0] for desc in cursor.description]
        rows    = cursor.fetchall()
        cursor.close()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=columns)

    # psycopg2 devuelve decimal.Decimal para columnas NUMERIC de PostgreSQL.
    # numpy y sklearn requieren float nativo — convertimos aquí, igual que
    # SQLAlchemy lo hace automáticamente en data_extraction.py.
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Clustering: %d transacciones ALTO RIESGO cargadas.", len(df))
    return df


# ──────────────────────────────────────────────────────────────────
# GENERACIÓN AUTOMÁTICA DE LABELS
# ──────────────────────────────────────────────────────────────────

def _generate_label(centroid: dict, high_amount_threshold: float) -> str:
    """
    Genera un label legible para un cluster a partir de su centroid.

    Combina características dominantes en un texto descriptivo:
    monto (alto/bajo), horario (nocturno/diurno) y distancia (lejano/cercano).
    """
    parts = []

    # Monto
    if centroid["amt"] >= high_amount_threshold:
        parts.append("alto monto")
    else:
        parts.append("bajo monto")

    # Horario
    h = centroid["hour"]
    if h >= _NIGHT_HOURS[0] or h < _NIGHT_HOURS[1]:
        parts.append("nocturno")
    elif 6 <= h < 12:
        parts.append("matutino")
    else:
        parts.append("diurno")

    # Distancia
    if centroid["distance_km"] > 200:
        parts.append("distancia lejana")
    elif centroid["distance_km"] < 20:
        parts.append("distancia cercana")

    return "Fraude " + ", ".join(parts)


# ──────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────

def compute_clusters(request: ClusteringRequest) -> ClusteringResponse:
    """
    Ejecuta el análisis de clustering completo.

    1. Carga transacciones ALTO RIESGO desde BD.
    2. Valida mínimo de muestras.
    3. Normaliza con StandardScaler.
    4. Aplica K-Means (random_state=42 para reproducibilidad).
    5. Interpreta centroids y genera labels.
    6. Calcula top_category y top_state por moda del cluster.
    7. Devuelve ClusteringResponse serializable.
    """
    run_ts = datetime.now().isoformat()
    logger.info("Iniciando clustering (K=%d, lookback=%s dias).", request.n_clusters, request.lookback_days)

    # ── 1. Extraccion de datos ────────────────────────────────────
    df = _load_fraud_data(request.lookback_days)

    total_frauds = len(df)
    if total_frauds < request.min_samples:
        logger.warning(
            "Clustering abortado: solo %d muestras (minimo %d).",
            total_frauds, request.min_samples,
        )
        return ClusteringResponse(
            profiles=[],
            total_frauds_analyzed=total_frauds,
            n_clusters_used=0,
            run_date=run_ts,
            message=f"Insufficient samples: {total_frauds} < {request.min_samples}",
        )

    # Ajustar K si hay menos muestras que clusters pedidos
    n_clusters = min(request.n_clusters, total_frauds // 10)
    n_clusters = max(n_clusters, 2)

    # ── 2. Preparacion de features ────────────────────────────────
    df_features = df[FEATURES].dropna()
    categories_series = df.loc[df_features.index, "category"]
    states_series     = df.loc[df_features.index, "state"]

    X = df_features.values.astype(float)

    # ── 3. Normalizacion ──────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 4. K-Means ────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels_arr = kmeans.fit_predict(X_scaled)

    # Centroids en espacio original (desnormalizados)
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

    # Umbral de monto alto = percentil 75 del dataset analizado
    high_amt_threshold = float(np.percentile(df_features["amt"].values, _HIGH_AMOUNT_QUANTILE * 100))

    # ── 5. Construccion de perfiles ───────────────────────────────
    profiles: List[ClusterProfile] = []

    for k in range(n_clusters):
        mask = labels_arr == k
        fraud_count = int(mask.sum())
        if fraud_count == 0:
            continue

        centroid = {
            feat: float(centroids_original[k, i])
            for i, feat in enumerate(FEATURES)
        }

        # Moda de categoria y estado para este cluster
        cluster_cats   = categories_series.iloc[mask]
        cluster_states = states_series.iloc[mask]
        top_category   = cluster_cats.mode().iloc[0] if not cluster_cats.dropna().empty else None
        top_state      = cluster_states.mode().iloc[0] if not cluster_states.dropna().empty else None

        label = _generate_label(centroid, high_amt_threshold)

        profiles.append(ClusterProfile(
            cluster_id      = k,
            label           = label,
            fraud_count     = fraud_count,
            pct_of_total    = round(fraud_count / total_frauds * 100, 2),
            avg_amount      = round(centroid["amt"], 2),
            avg_hour        = round(centroid["hour"], 1),
            avg_age         = round(centroid["age"], 1),
            avg_distance_km = round(centroid["distance_km"], 2),
            avg_city_pop    = round(centroid["city_pop"], 0),
            top_category    = top_category,
            top_state       = top_state,
        ))

    # Ordenar de mayor a menor (más fraudes primero)
    profiles.sort(key=lambda p: p.fraud_count, reverse=True)

    logger.info(
        "Clustering completado: %d clusters, %d fraudes analizados.",
        n_clusters, total_frauds,
    )

    return ClusteringResponse(
        profiles               = profiles,
        total_frauds_analyzed  = total_frauds,
        n_clusters_used        = n_clusters,
        run_date               = run_ts,
        message                = f"OK: {n_clusters} clusters generados sobre {total_frauds} fraudes.",
    )
