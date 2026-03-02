# src/fraude/fraude_schema.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class FraudTrainingSample(BaseModel):
    """Una muestra de transacción para entrenamiento de fraude."""
    
    # Features numéricas continuas
    amt: float = Field(..., description="Monto de la transacción")
    city_pop: int = Field(..., description="Población de la ciudad")
    
    # Features categóricas (se recibirán como strings)
    category: str = Field(..., description="Categoría del comercio")
    gender: str = Field(..., description="Género del cliente (M/F)")
    job: str = Field(..., description="Profesión del cliente")
    
    # Features de geolocalización
    lat: float = Field(..., description="Latitud del cliente")
    long: float = Field(..., description="Longitud del cliente")
    merch_lat: float = Field(..., description="Latitud del comercio")
    merch_long: float = Field(..., description="Longitud del comercio")
    
    # Features temporales
    trans_date_trans_time: str = Field(..., description="Timestamp de la transacción")
    dob: str = Field(..., description="Fecha de nacimiento del cliente")
    
    # Label y peso
    is_fraud: int = Field(..., description="Label: 1=fraude, 0=legítimo")
    sample_weight: float = Field(1.0, description="Peso temporal por decay")


class TrainingRequest(BaseModel):
    """
    Request para entrenar el modelo de fraude.
    
    La API extrae los datos automáticamente de la base de datos
    en el rango de fechas especificado, aplicando sampling balanceado.
    """
    start_date: str = Field(..., description="Fecha inicial (YYYY-MM-DD)", example="2025-09-01")
    end_date: str = Field(..., description="Fecha final (YYYY-MM-DD)", example="2026-02-12")
    optuna_trials: int = Field(30, description="Número de trials de Optuna para optimización")
    undersampling_ratio: int = Field(4, description="Ratio de legítimas por fraude (default: 4:1)")
    
    # Opcional: ID del audit record creado por Java (si viene desde Java)
    audit_id: int | None = Field(None, description="ID del registro de auditoría creado por Java")
    triggered_by: str = Field("manual", description="Quién disparó el entrenamiento: manual, scheduled, drift_detected")


class OptunaResult(BaseModel):
    """Resultado de la optimización con Optuna."""
    best_trial_number: int
    best_f1_score: float
    best_params: Dict[str, Any]


class TrainingMetrics(BaseModel):
    """Métricas del modelo de fraude entrenado."""
    auc_roc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    optimal_threshold: float = Field(..., description="Threshold que maximiza Recall >= 95%")
    training_time_sec: float


class TrainingResponse(BaseModel):
    """Respuesta del entrenamiento de fraude."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    metrics: TrainingMetrics
    optuna_result: OptunaResult
    model_base64: str = Field(..., description="Modelo híbrido serializado en base64")
    model_config_dict: Dict[str, Any] = Field(..., description="Configuración del modelo híbrido")
    promotion_status: str = Field(..., description="Estado de promoción: PROMOTED o REJECTED")
    total_samples: int
    train_samples: int
    test_samples: int
    class_distribution: Dict[str, int] = Field(..., description="Distribución de clases en dataset balanceado")
    fraud_ratio_balanced: float = Field(..., description="Ratio de fraude después de sampling")
