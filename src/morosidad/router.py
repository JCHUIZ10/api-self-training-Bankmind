# src/morosidad/router.py
import logging
from fastapi import APIRouter, HTTPException

from morosidad.morosidad_schema import TrainingRequest, TrainingResponse
from morosidad.training_service import entrenar_modelo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/morosidad", tags=["Morosidad - Self Training"])


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Entrena un modelo candidato de morosidad con los datos proporcionados.

    El backend Java envía el dataset completo (extraído de la vista materializada).
    Esta API se encarga de:
    1. Optimizar hiperparámetros con Optuna
    2. Entrenar un ensemble (XGBoost + LightGBM + RF)
    3. Calcular métricas de evaluación
    4. Retornar el modelo serializado + métricas
    """
    if not request.samples:
        raise HTTPException(status_code=400, detail="El dataset no puede estar vacío")

    if len(request.samples) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset muy pequeño ({len(request.samples)} muestras). Se requieren al menos 100."
        )

    logger.info(f"📡 Solicitud de entrenamiento recibida: {len(request.samples)} muestras, "
                f"{request.optuna_trials} trials Optuna")

    try:
        response = entrenar_modelo(request)
        logger.info(f"✅ Entrenamiento exitoso. AUC: {response.metrics.auc_roc}")
        return response
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en entrenamiento: {str(e)}")
