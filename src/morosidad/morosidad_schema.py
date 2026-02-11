# src/morosidad/morosidad_schema.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TrainingSample(BaseModel):
    """Una muestra de entrenamiento: 24 features + label + peso temporal."""

    # Datos de cuenta
    LIMIT_BAL: float = Field(..., description="Límite de crédito")

    # Datos demográficos
    SEX: int = Field(..., description="Género (1=Masculino, 2=Femenino)")
    EDUCATION: int = Field(..., description="Nivel educativo (1-4)")
    MARRIAGE: int = Field(..., description="Estado civil (1-3)")
    AGE: int = Field(..., description="Edad")

    # Estado de pago mensual (6 meses, más reciente primero)
    PAY_0: int = Field(..., description="Estado de pago mes más reciente")
    PAY_2: int = Field(..., description="Estado de pago mes -2")
    PAY_3: int = Field(..., description="Estado de pago mes -3")
    PAY_4: int = Field(..., description="Estado de pago mes -4")
    PAY_5: int = Field(..., description="Estado de pago mes -5")
    PAY_6: int = Field(..., description="Estado de pago mes -6")

    # Montos facturados (6 meses)
    BILL_AMT1: float = Field(..., description="Monto facturado mes -1")
    BILL_AMT2: float = Field(..., description="Monto facturado mes -2")
    BILL_AMT3: float = Field(..., description="Monto facturado mes -3")
    BILL_AMT4: float = Field(..., description="Monto facturado mes -4")
    BILL_AMT5: float = Field(..., description="Monto facturado mes -5")
    BILL_AMT6: float = Field(..., description="Monto facturado mes -6")

    # Montos pagados (6 meses)
    PAY_AMT1: float = Field(..., description="Monto pagado mes -1")
    PAY_AMT2: float = Field(..., description="Monto pagado mes -2")
    PAY_AMT3: float = Field(..., description="Monto pagado mes -3")
    PAY_AMT4: float = Field(..., description="Monto pagado mes -4")
    PAY_AMT5: float = Field(..., description="Monto pagado mes -5")
    PAY_AMT6: float = Field(..., description="Monto pagado mes -6")

    # Tasa de utilización
    UTILIZATION_RATE: float = Field(..., description="Tasa de utilización del crédito")

    # Label y peso
    default_payment_next_month: int = Field(..., description="Label: 1=moroso, 0=no moroso")
    sample_weight: float = Field(1.0, description="Peso temporal por decay")


class TrainingRequest(BaseModel):
    """Request para entrenar el modelo."""
    samples: List[TrainingSample]
    optuna_trials: int = Field(30, description="Número de trials de Optuna")


class TrainingMetrics(BaseModel):
    """Métricas del modelo entrenado."""
    auc_roc: float
    ks_statistic: float
    gini_coefficient: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time_sec: float


class OptunaResult(BaseModel):
    """Resultado de la optimización con Optuna."""
    trial_number: int
    objective_value: float
    metric_optimized: str
    best_params: Dict[str, Any]


class TrainingResponse(BaseModel):
    """Respuesta del entrenamiento."""
    metrics: TrainingMetrics
    optuna_result: OptunaResult
    model_base64: str = Field(..., description="Modelo serializado en base64")
    assembly_config: Dict[str, Any] = Field(..., description="Configuración del ensemble")
    total_samples: int
    train_samples: int
    test_samples: int
    class_distribution: Dict[str, int]
    scale_pos_weight: float
