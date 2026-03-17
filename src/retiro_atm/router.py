# src/retiro_atm/router.py
import logging
import uuid
from fastapi import APIRouter, HTTPException,BackgroundTasks

from retiro_atm.schemas import TrainingRequest, TrainingResponse
from retiro_atm.self_train import training_service
from retiro_atm.generated.synthetic_data_service import ejecutar_sync
from retiro_atm.monitoring.orquestador_monitoreo import ejecutar_monitoreo
from retiro_atm.database import init_db, get_engine 


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retiro-atm", tags=["Retiro ATM - Self Training"])

@router.on_event("startup")
def startup_event():
    init_db()

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


@router.post("/new-data")
def generated_new_data(fecha_objetivo: str, background_tasks: BackgroundTasks):
    """
    Endpoint para iniciar el proceso de generación de nuevos datos ATM.
    - fecha_objetivo: Fecha objetivo para la generación de datos (default: fecha actual) - YYYY-MM-DD
    """
    try:
        engine = get_engine()
        
        task_id = str(uuid.uuid4()) # ID único para esta tarea
        logger.info(f"📡 Solicitud recibida. Task ID: {task_id}")
        background_tasks.add_task(ejecutar_sync, fecha_objetivo, engine)

        return {
            "status": "accepted",
            "task_id": task_id,
            "message": f"Proceso iniciado para la fecha {fecha_objetivo}."
        }
    except Exception as e:
        logger.error(f"❌ Error durante la generación de nuevos datos: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error en generación de nuevos datos ATM: {str(e)}",
        )

@router.post("/monitoring")
def monitoring_data(background_tasks: BackgroundTasks):
    """
    Endpoint para iniciar el proceso de monitoreo de datos ATM.
    """
    try:
        engine = get_engine()

        background_tasks.add_task(ejecutar_monitoreo, engine)
        return {
            "status": "accepted",
            "message": f"Proceso iniciado para el monitoreo."
        }
    except Exception as e:
        logger.error(f"❌ Error durante el monitoreo: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error en el monitoreo de datos: {str(e)}",
        )