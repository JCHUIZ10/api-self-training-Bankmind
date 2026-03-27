# src/fraude/router.py
import asyncio
import logging
from fastapi import APIRouter, HTTPException

from fraude.schemas.fraude import TrainingRequest, TrainingResponse
from fraude.core.training.training_pipeline import entrenar_modelo
from fraude.core.drift import (
    DriftCalculationRequest,
    DriftCalculationResponse,
    calculate_drift,
)
from fraude.schemas.clustering import ClusteringRequest, ClusteringResponse
from fraude.core.clustering import compute_clusters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fraude", tags=["Fraude - Self Training"])


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Entrena un modelo candidato de fraude extrayendo datos automáticamente de la BD.

    Pipeline de autoentrenamiento:
    1. Valida rango de fechas
    2. Extrae datos de operational_transactions con sampling balanceado
    3. Feature engineering (age, hour, distance_km)
    4. Encoding y scaling
    5. IsolationForest para generar anomaly_score
    6. Optimización de hiperparámetros con Optuna (maximiza F1-Score)
    7. XGBoost con mejores parámetros encontrados
    8. Threshold optimization (maximiza Precision con Recall >= 95%)
    9. Serializa modelo y retorna con métricas

    Nota: El entrenamiento puede tardar varios minutos.
    Se ejecuta en un thread pool para no bloquear el event loop de FastAPI.
    """
    logger.info("[REQUEST] Solicitud de autoentrenamiento recibida")
    logger.info("   Período: %s → %s", request.start_date, request.end_date)
    logger.info("   Trials Optuna: %s", request.optuna_trials)
    logger.info("   Ratio undersampling: %s:1", request.undersampling_ratio)

    try:
        # Ejecutar en thread pool: entrenar_modelo es síncrono y tarda minutos.
        # run_in_executor evita bloquear el event loop durante todo el entrenamiento.
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, entrenar_modelo, request)

        logger.info("[OK] Autoentrenamiento completado exitosamente")
        logger.info("   F1: %s | AUC: %s", response.metrics.f1_score, response.metrics.auc_roc)
        logger.info("   Threshold: %s", response.metrics.optimal_threshold)
        logger.info("   Best trial: #%s", response.optuna_result.best_trial_number)
        logger.info("   Status: %s", response.promotion_status)
        return response

    except Exception as exc:
        # logger.exception incluye el stacktrace completo — no se expone al cliente
        logger.exception("[ERROR] Error durante el autoentrenamiento: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Error interno durante el autoentrenamiento. Revisa los logs del servidor.",
        )


@router.post("/drift/calculate", response_model=DriftCalculationResponse)
async def calculate_feature_drift(request: DriftCalculationRequest):
    """
    Calcula el PSI por feature comparando datos recientes con el baseline del CHAMPION.

    Llamado por el Scheduler de Java cada 24 horas para detectar drift proactivo.

    Args:
        start_date: Fecha inicio de la ventana a analizar ('YYYY-MM-DD')
        end_date:   Fecha fin de la ventana a analizar ('YYYY-MM-DD')
        persist:    Si True (default), guarda resultados en model_feature_drift

    Returns:
        DriftCalculationResponse con PSI por feature, flags de alerta y features críticos.
        - has_critical_drift: True si algún feature supera PSI > 0.25
        - critical_features:  Lista de features con drift severo
    """
    logger.info("[DRIFT] Solicitud de cálculo PSI: %s → %s", request.start_date, request.end_date)
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, calculate_drift, request)
        logger.info(
            "[DRIFT] PSI calculado para %d features. Críticos: %s",
            len(response.features),
            response.critical_features or "ninguno",
        )
        return response

    except Exception as exc:
        logger.exception("[DRIFT ERROR] %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Error calculando drift. Revisa los logs del servidor.",
        )


@router.post("/clustering/compute", response_model=ClusteringResponse)
async def compute_fraud_clusters(request: ClusteringRequest):
    """
    Ejecuta K-Means clustering sobre transacciones ALTO RIESGO.

    Identifica perfiles automáticos de defraudadores agrupando
    las transacciones fraudulentas por características similares
    (monto, hora, edad, distancia, ciudad).

    Llamado por el Scheduler de Java semanalmente (Lunes 03:00 AM)
    o manualmente desde el Dashboard.

    Args:
        n_clusters:    Número de grupos a identificar (default 3).
        min_samples:   Mínimo de muestras para ejecutar el análisis (default 30).
        lookback_days: Si se especifica, limita a los últimos N días.

    Returns:
        ClusteringResponse con lista de ClusterProfile (perfiles de defraudadores)
        ordenados de mayor a menor frecuencia.
    """
    logger.info(
        "[CLUSTERING] Solicitud recibida: K=%d, min_samples=%d, lookback=%s",
        request.n_clusters, request.min_samples, request.lookback_days,
    )
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, compute_clusters, request)
        logger.info(
            "[CLUSTERING] Completado: %d clusters, %d fraudes analizados.",
            response.n_clusters_used, response.total_frauds_analyzed,
        )
        return response
    except Exception as exc:
        logger.exception("[CLUSTERING ERROR] %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Error ejecutando clustering. Revisa los logs del servidor.",
        )
