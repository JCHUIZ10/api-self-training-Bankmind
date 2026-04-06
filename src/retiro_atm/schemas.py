# src/retiro_atm/schemas.py
from datetime import datetime, date
from typing import Dict, Any, List, Optional

from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Date, JSON, Numeric, Boolean, DateTime

from retiro_atm.database import Base


# ═══════════════════════════════════════════════════════════
# MODELOS SQLALCHEMY (Persistencia en PostgreSQL)
# ═══════════════════════════════════════════════════════════

class DatasetWithdrawalPrediction(Base):
    """Registro del dataset utilizado para un entrenamiento."""
    __tablename__ = "dataset_withdrawal_prediction"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    start_date = Column(Date)
    end_date = Column(Date)
    count_total = Column(Integer)
    count_train = Column(Integer)
    count_test = Column(Integer)
    features = Column(JSON)
    target = Column(String)
    created_at = Column(DateTime, default=datetime.now)


class SelfTrainingAuditWithdrawalModel(Base):
    """Auditoría de cada ciclo de autoentrenamiento."""
    __tablename__ = "self_training_audit_withdrawal_model"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    model_name = Column(String)
    mae = Column(Numeric(precision=15, scale=3))
    mape = Column(Numeric(precision=15, scale=3))
    rmse = Column(Numeric(precision=15, scale=3))
    margin_improvement = Column(Numeric(precision=10, scale=4))
    training_duration_minutes = Column(Integer)
    start_training = Column(DateTime)
    end_training = Column(DateTime)
    hyperparameters = Column(JSON)
    psi_baseline = Column(JSON)
    is_production = Column(Boolean)
    compared_to_model = Column(Integer)
    id_dataset_withdrawal_prediction = Column(Integer)


class WithdrawalModel(Base):
    """Modelo de producción activo con metadatos estadísticos."""
    __tablename__ = "withdrawal_models"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    confidence_level = Column(Numeric(precision=5, scale=2), nullable=False)
    end_date = Column(Date, nullable=True)
    is_active = Column(Boolean, nullable=False)
    margin = Column(Numeric(precision=15, scale=3), nullable=False)
    sigma = Column(Numeric(precision=15, scale=3), nullable=False)
    start_date = Column(Date, nullable=False)
    t_crit = Column(Numeric(precision=6, scale=4), nullable=False)
    id_self_training_audit_withdrawal_model = Column(Integer, unique=True, nullable=False)
    importances_features = Column(JSON, nullable=False)


# ═══════════════════════════════════════════════════════════
# MODELOS PYDANTIC (Request / Response de la API)
# ═══════════════════════════════════════════════════════════

class TrainingRequest(BaseModel):
    """Configuración para iniciar el autoentrenamiento."""
    optuna_trials: int = 50
    tolerancia_mape: float = 0.05
    dias_particion_test: int = 60
    dias_particion_val: int = 15


class TrainingMetrics(BaseModel):
    """Métricas de evaluación del modelo."""
    mae: float
    mape: float
    rmse: float
    training_time_sec: float


class ConfidenceInterval(BaseModel):
    """Intervalo de confianza de los residuos."""
    lower_bound: float
    upper_bound: float
    media_residuos: float
    sigma: float
    margin_error: float
    confidence_level: float = 95.0
    t_crit: float


class TrainingResponse(BaseModel):
    """Respuesta completa del pipeline de autoentrenamiento."""
    # Métricas del nuevo modelo
    metrics_challenger: TrainingMetrics
    # Métricas del modelo en producción (None si no existe)
    metrics_champion: Optional[TrainingMetrics] = None

    # Resultado de Optuna
    best_params: Dict[str, Any]
    n_trials: int

    # Info del dataset
    total_samples: int
    train_samples: int
    test_samples: int

    # Importancia de features
    feature_importances: Dict[str, float]

    # Intervalo de confianza
    confidence_interval: Optional[ConfidenceInterval] = None

    # Decisión de promoción
    deployment_status: str  # NEW_CHAMPION, KEEP_CHAMPION, UPLOAD_FAILED
    version_tag: Optional[str] = None
    margin_improvement: Optional[float] = None

    # Verificación DagsHub
    dagshub_verified: bool = False
