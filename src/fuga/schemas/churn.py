# src/fuga/schemas/churn.py
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class TrainingRequest(BaseModel):
    """Request para disparar el auto-entrenamiento del modelo de churn."""
    triggered_by:  str          = Field("manual", description="manual | scheduled | drift_detected | performance_monitor")
    audit_id:      Optional[int] = Field(None,    description="ID de auditoría pre-creado por Java (opcional)")
    optuna_trials: int           = Field(20,       description="Número de trials Optuna para búsqueda de hiperparámetros (min 5)", ge=5)


class TrainingMetrics(BaseModel):
    """Métricas del modelo de churn entrenado."""
    accuracy:          float
    f1_score:          float
    precision:         float
    recall:            float
    auc_roc:           float
    training_time_sec: float


class TrainingResponse(BaseModel):
    """Respuesta del auto-entrenamiento de churn."""
    model_config = ConfigDict(protected_namespaces=())

    metrics:           TrainingMetrics
    best_params:       Dict[str, Any]
    promotion_status:  str   = Field(..., description="PROMOTED | REJECTED | PERSISTENCE_ERROR")
    promotion_reason:  str
    total_samples:     int
    train_samples:     int
    test_samples:      int
    class_distribution: Dict[str, int]
    churn_ratio:       float = Field(..., description="Ratio de churn en el dataset original")
    model_version:     str
    champion_metrics:  Optional[Dict[str, float]] = None
    dagshub_verified:  bool = False
    dagshub_url:       Optional[str] = None
    mlflow_run_id:     Optional[str] = None
