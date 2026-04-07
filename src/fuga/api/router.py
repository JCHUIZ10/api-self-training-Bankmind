# src/fuga/api/router.py
import asyncio
import logging

from fastapi import APIRouter, HTTPException

from fuga.schemas.churn import TrainingRequest, TrainingResponse
from fuga.core.training.training_pipeline import entrenar_modelo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fuga", tags=["Fuga (Churn) - Self Training"])


@router.post("/train", response_model=TrainingResponse)
async def train_churn_model(request: TrainingRequest):
    """
    Dispara el pipeline completo de auto-entrenamiento del modelo de Churn.

    Pipeline:
    1. Extrae datos de account_details (PostgreSQL)
    2. Feature engineering + encoding
    3. SMOTE para balanceo de clases
    4. GridSearchCV (XGBoost, 3-fold, scoring=roc_auc)
    5. Evaluacion del challenger
    6. Champion/Challenger (Score: AUC*40% + F1*30% + Recall*30%)
    7. Si el challenger gana: upload a DagsHub + hot-reload al servidor principal
    8. Persistencia en churn_models, dataset_churn_prediction, self_training_audit_churn

    El entrenamiento puede tardar varios minutos.
    Se ejecuta en un thread pool para no bloquear el event loop.
    """
    logger.info("[Fuga] Solicitud de auto-entrenamiento recibida (triggered_by=%s)", request.triggered_by)
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, entrenar_modelo, request)
        logger.info(
            "[Fuga] Entrenamiento completado — status=%s, AUC=%.4f, F1=%.4f",
            response.promotion_status,
            response.metrics.auc_roc,
            response.metrics.f1_score,
        )
        return response
    except Exception as exc:
        logger.exception("[Fuga] Error durante el auto-entrenamiento: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Error interno durante el auto-entrenamiento de churn. Revisa los logs.",
        )
