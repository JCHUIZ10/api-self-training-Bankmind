# src/fraude/drift_service.py
"""
Servicio de Cálculo de Drift / PSI por Feature.

Responsabilidad:
  - Dada una ventana de transacciones recientes, calcular el PSI
    de cada feature numérico contra la distribución baseline del CHAMPION.
  - El baseline se lee del campo model_config.baseline_distributions
    guardado en la tabla fraud_models.
  - Persiste los resultados en model_feature_drift (leída por el Frontend).

Flujo de uso:
  Java Scheduler  →  POST /fraude/drift/calculate  →  drift_service.py
  drift_service  →  BD (fraud_models → baseline) + BD (operational_transactions → datos recientes)
  drift_service  →  Persiste en model_feature_drift
  drift_service  →  Retorna lista de PSI por feature
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel
from sqlalchemy.orm import Session

from fraude.data.db_config import get_db_session
from fraude.data.data_extraction import get_raw_transactions

logger = logging.getLogger(__name__)


def _haversine_np(lon1, lat1, lon2, lat2):
    """Calcula distancia en km entre dos puntos geográficos (Haversine).
    Definida aquí para evitar importaciones circulares con training_service."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# =============================================================================
#  SCHEMAS
# =============================================================================

class FeatureDriftResult(BaseModel):
    feature_name: str
    psi_value: float
    drift_category: str          # 'LOW' | 'MODERATE' | 'HIGH'
    measured_at: str             # ISO timestamp

class DriftCalculationResponse(BaseModel):
    id_champion_model: int
    start_date: str
    end_date: str
    features: List[FeatureDriftResult]
    has_critical_drift: bool          # True si algún feature > 0.25
    critical_features: List[str]      # Features por encima del umbral

class DriftCalculationRequest(BaseModel):
    start_date: str   # 'YYYY-MM-DD'  →  ventana a comparar contra baseline
    end_date: str
    persist: bool = True             # Si True, guarda en model_feature_drift

# =============================================================================
#  LÓGICA CORE
# =============================================================================

# Umbrales estándar de PSI para fraude bancario
PSI_LOW      = 0.10   # Sin drift
PSI_MODERATE = 0.25   # Vigilar; dispara alerta amarilla
# > 0.25 → HIGH, dispara entrenamiento


def _psi_category(psi: float) -> str:
    if psi < PSI_LOW:
        return "LOW"
    elif psi < PSI_MODERATE:
        return "MODERATE"
    return "HIGH"


def _calculate_psi_for_feature(
    baseline_pct: List[float],
    baseline_bins: List[float],
    current_data: np.ndarray,
) -> float:
    """
    Calcula el PSI de un feature dado.

    PSI = Σ (current_pct_i - baseline_pct_i) * ln(current_pct_i / baseline_pct_i)

    Args:
        baseline_pct:  Distribución de referencia (lista de %, suma ≈ 100).
        baseline_bins: Bordes de los bins del baseline (longitud = len(baseline_pct) + 1).
        current_data:  Array con los valores actuales del feature.

    Returns:
        Valor PSI (float ≥ 0).
    """
    if len(current_data) == 0:
        logger.warning("current_data vacío, PSI = 0")
        return 0.0

    # Mapear datos actuales a los mismos bins del baseline
    # np.histogram requiere que los edges sean estrictamente crecientes;
    # los bins de percentil pueden tener duplicados → los limpiamos.
    bins = np.unique(baseline_bins)
    if len(bins) < 2:
        logger.warning("Bins inválidos (todos iguales), PSI = 0")
        return 0.0

    current_counts, _ = np.histogram(current_data, bins=bins)
    current_total = current_counts.sum()
    if current_total == 0:
        return 0.0

    current_pct = current_counts / current_total * 100.0

    # Ajustar tamaños si difieren (puede pasar por duplicados en bins)
    min_len = min(len(baseline_pct), len(current_pct))
    base  = np.array(baseline_pct[:min_len], dtype=float)
    curr  = current_pct[:min_len]

    # Evitar división por cero / log(0) con epsilon
    eps   = 1e-6
    base  = np.where(base  < eps, eps, base)
    curr  = np.where(curr  < eps, eps, curr)

    psi = float(np.sum((curr - base) * np.log(curr / base)))
    return round(abs(psi), 6)    # PSI siempre positivo


def _get_champion_model_config(session: Session) -> Optional[Dict]:
    """
    Lee el model_config del CHAMPION activo (contiene baseline_distributions).
    Retorna el dict parseado o None si no hay champion o falta el baseline.
    """
    from fraude.data import model_registry
    champion = model_registry.get_current_champion(session)
    if not champion:
        logger.warning("No hay modelo CHAMPION activo. PSI no disponible.")
        return None, None

    if not champion.model_config:
        logger.warning("Champion sin model_config. PSI no disponible.")
        return None, None

    try:
        config = json.loads(champion.model_config) if isinstance(champion.model_config, str) else champion.model_config
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error parseando model_config del champion: {e}")
        return None, None

    baseline = config.get("baseline_distributions")
    if not baseline:
        logger.warning("model_config no contiene baseline_distributions. Entrena un nuevo Champion para habilitarlo.")
        return None, None

    return champion, baseline


def _persist_drift_results(
    session: Session,
    id_model: int,
    drift_results: List[FeatureDriftResult],
) -> None:
    """Persiste los resultados de PSI en model_feature_drift."""
    from fraude.data.db_models import ModelFeatureDrift as ModelFeatureDriftORM
    for result in drift_results:
        record = ModelFeatureDriftORM(
            id_model=id_model,
            feature_name=result.feature_name,
            psi_value=result.psi_value,
            drift_category=result.drift_category,
        )
        session.add(record)
    session.flush()
    logger.info(f"💾 {len(drift_results)} registros de drift guardados en BD")


# =============================================================================
#  FUNCIÓN PÚBLICA
# =============================================================================

def calculate_drift(request: DriftCalculationRequest) -> DriftCalculationResponse:
    """
    Calcula el PSI por feature comparando los datos del período indicado
    contra la distribución baseline del modelo CHAMPION.

    Argumentos:
        request.start_date / end_date: Ventana de datos a comparar.
        request.persist: Si True, persiste en model_feature_drift.

    Retorna DriftCalculationResponse con PSI por feature y flags de alerta.
    """
    logger.info(f"🔬 Calculando Drift PSI: {request.start_date} → {request.end_date}")
    now_str = datetime.now().isoformat()

    with get_db_session() as session:
        # 1. Obtener baseline del CHAMPION
        champion, baseline_distributions = _get_champion_model_config(session)
        if champion is None:
            return DriftCalculationResponse(
                id_champion_model=-1,
                start_date=request.start_date,
                end_date=request.end_date,
                features=[],
                has_critical_drift=False,
                critical_features=[],
            )

        # ⚠️ Guardar el ID antes de que la sesión se cierre
        # (evita DetachedInstanceError al acceder champion.id_model fuera del with)
        champion_id = int(champion.id_model)

        # 2. Extraer datos recientes de BD
        try:
            df_current = get_raw_transactions(
                start_date=request.start_date,
                end_date=request.end_date,
            )
        except Exception as e:
            logger.error(f"Error extrayendo transacciones recientes: {e}")
            raise

        # 3. Feature engineering mínimo para tener las mismas features
        df_current['trans_date_trans_time'] = pd.to_datetime(df_current['trans_date_trans_time'])
        df_current['dob'] = pd.to_datetime(df_current['dob'])
        # relativedelta calcula años exactos, considerando bisiestos
        df_current['age'] = df_current.apply(
            lambda r: relativedelta(r['trans_date_trans_time'], r['dob']).years, axis=1
        )
        df_current['hour'] = df_current['trans_date_trans_time'].dt.hour
        # distance_km: se calcula aquí directamente con haversine para evitar
        # importaciones circulares con training_service
        if 'distance_km' not in df_current.columns:
            try:
                geo_cols = ['long', 'lat', 'merch_long', 'merch_lat']
                if all(c in df_current.columns for c in geo_cols):
                    df_current['distance_km'] = _haversine_np(
                        df_current['long'], df_current['lat'],
                        df_current['merch_long'], df_current['merch_lat']
                    )
                    logger.info("📍 distance_km calculado correctamente en drift_service")
                else:
                    missing = [c for c in geo_cols if c not in df_current.columns]
                    logger.warning(f"No se puede calcular distance_km — columnas faltantes: {missing}")
            except Exception as exc:
                logger.error(f"Error calculando distance_km en drift_service: {exc}")

        # 4. Calcular PSI por feature
        drift_results: List[FeatureDriftResult] = []
        critical_features: List[str] = []

        for feature_name, baseline_data in baseline_distributions.items():
            try:
                if feature_name not in df_current.columns:
                    logger.warning(f"Feature '{feature_name}' no disponible en datos actuales, omitiendo.")
                    continue

                current_values = df_current[feature_name].dropna().values
                psi = _calculate_psi_for_feature(
                    baseline_pct=baseline_data["pct"],
                    baseline_bins=baseline_data["bins"],
                    current_data=current_values,
                )
                category = _psi_category(psi)
                if category == "HIGH":
                    critical_features.append(feature_name)

                drift_results.append(FeatureDriftResult(
                    feature_name=feature_name,
                    psi_value=psi,
                    drift_category=category,
                    measured_at=now_str,
                ))
                logger.info(f"   [{category}] {feature_name}: PSI = {psi:.5f}")

            except Exception as e:
                logger.warning(f"No se pudo calcular PSI para '{feature_name}': {e}")

        # 5. Persistir si se solicita
        if request.persist and drift_results:
            try:
                _persist_drift_results(session, champion_id, drift_results)
                session.commit()
            except Exception as e:
                logger.error(f"Error persistiendo drift: {e}")
                session.rollback()

    has_critical = len(critical_features) > 0
    logger.info(
        f"✅ PSI calculado para {len(drift_results)} features. "
        f"Críticos: {critical_features or 'ninguno'}"
    )

    return DriftCalculationResponse(
        id_champion_model=champion_id,
        start_date=request.start_date,
        end_date=request.end_date,
        features=drift_results,
        has_critical_drift=has_critical,
        critical_features=critical_features,
    )