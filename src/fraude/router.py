# src/fraude/router.py
import logging
from fastapi import APIRouter, HTTPException

from fraude.fraude_schema import TrainingRequest, TrainingResponse
from fraude.training_service import entrenar_modelo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fraude", tags=["Fraude - Self Training"])


@router.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Entrena un modelo candidato de fraude extrayendo datos automáticamente de la BD.
    
    Pipeline de autoentrenamiento:
    1. Valida rango de fechas
    2. Extrae datos de operational_transactions con sampling balanceado:
       - 100% de fraudes
       - N legítimas por fraude (configurable, default 4:1)
    3. Feature engineering (age, hour, distance_km)
    4. Encoding y scaling
    5. IsolationForest para generar anomaly_score
    6. Optimización de hiperparámetros con Optuna (maximiza F1-Score)
    7. XGBoost con mejores parámetros encontrados
    8. Threshold optimization (maximiza Precision con Recall >= 95%)
    9. Serializa modelo y retorna con métricas
    
    Args:
        request: TrainingRequest con start_date, end_date, optuna_trials, undersampling_ratio
    
    Returns:
        TrainingResponse con modelo serializado, métricas, y promotion_status
    """
    logger.info(f"[REQUEST] Solicitud de autoentrenamiento recibida")
    logger.info(f"   Período: {request.start_date} → {request.end_date}")
    logger.info(f"   Trials Optuna: {request.optuna_trials}")
    logger.info(f"   Ratio undersampling: {request.undersampling_ratio}:1")
    
    
    try:
        response = entrenar_modelo(request)
        logger.info(f"[OK] Autoentrenamiento completado exitosamente")
        logger.info(f"   F1: {response.metrics.f1_score} | AUC: {response.metrics.auc_roc}")
        logger.info(f"   Threshold: {response.metrics.optimal_threshold}")
        logger.info(f"   Best trial: #{response.optuna_result.best_trial_number}")
        logger.info(f"   Status: {response.promotion_status}")
        return response
    except Exception as e:
        logger.error(f"[ERROR] Error durante el autoentrenamiento: {e}", exc_info=True)
        
        # Incluir traceback completo en la respuesta para debugging
        import traceback
        traceback_str = traceback.format_exc()
        logger.error(f"Traceback completo:\n{traceback_str}")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Error en autoentrenamiento: {str(e)}\n\nTraceback:\n{traceback_str}"
        )
