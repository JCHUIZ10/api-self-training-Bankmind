from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict, Any


class TrainingRequest(BaseModel):
    optuna_trials: Optional[int] = 30


class TrainingMetrics(BaseModel):
    auc_roc: float
    ks_statistic: float
    gini_coefficient: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time_sec: float


class OptunaResult(BaseModel):
    best_value: float
    best_params: Dict[str, Any]
    n_trials: int


class TrainingResponse(BaseModel):
    metrics: TrainingMetrics
    optuna_result: OptunaResult

    # Info del dataset (se almacena en dataset_info)
    total_samples: int
    train_samples: int
    test_samples: int

    # Distribuciones de referencia para PSI (se almacena en training_history)
    baseline_distributions: Dict[str, Dict[str, Any]]

    # Configuración del ensamble (se almacena en production_model_default)
    assembly_config: Optional[Dict[str, Any]] = None

    # Info de columnas usadas en el entrenamiento
    columns_info: Optional[List[Dict[str, Any]]] = None

    # Fecha más antigua del dataset usado
    dataset_start_date: Optional[str] = None

    # Verificación de integridad en DagsHub
    dagshub_verified: bool = False

    # Version tag del modelo entrenado (se usa en BD y DagsHub)
    version_tag: Optional[str] = None

    # Status de despliegue (controla flujo en backend)
    deployment_status: str = "UNKNOWN"  # NEW_CHAMPION, KEEP_CHAMPION, UPLOAD_FAILED, ERROR
