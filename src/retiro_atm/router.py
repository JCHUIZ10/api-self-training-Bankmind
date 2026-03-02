# src/retiro_atm/router.py
import logging
from fastapi import APIRouter, HTTPException

from retiro_atm.schemas import TrainingRequest, TrainingResponse
from retiro_atm import training_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retiro-atm", tags=["Retiro ATM - Self Training"])


@router.post("/train", response_model=TrainingResponse)
def train_model_endpoint(request: TrainingRequest):
    """
    Endpoint para iniciar el autoentrenamiento del modelo de retiro ATM.

    - **optuna_trials**: Número de trials para la optimización (default: 100)
    - **tolerancia_mape**: Tolerancia mínima de mejora MAPE (default: 0.05)
    - **dias_particion_test**: Días para el split de test (default: 60)
    - **dias_particion_val**: Días para el split de validación (default: 15)
    """
    logger.info(
        f"📡 Solicitud de entrenamiento ATM recibida "
        f"(trials={request.optuna_trials}, "
        f"tolerancia={request.tolerancia_mape})"
    )

    try:
        return training_service.ejecutar_autoentrenamiento(request)
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error en entrenamiento ATM: {str(e)}",
        )
