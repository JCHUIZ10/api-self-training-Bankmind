# src/morosidad/router.py
import logging
from fastapi import APIRouter, HTTPException

from morosidad.morosidad_schema import TrainingRequest, TrainingResponse
from morosidad import training_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/morosidad", tags=["Morosidad - Self Training"])

@router.post("/train", response_model=TrainingResponse)
def train_model_endpoint(request: TrainingRequest):
    """
    Endpoint para iniciar el auto-entrenamiento (Orquestación Python).
    """
    logger.info(f"📡 Solicitud de entrenamiento recibida (Trials={request.optuna_trials})")

    try:
        return training_service.ejecutar_autoentrenamiento(request)
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")
