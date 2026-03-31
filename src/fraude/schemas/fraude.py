# src/fraude/fraude_schema.py
from typing import List, Dict, Any, Optional
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
    sample_weight: float = Field(1.0, description="Peso temporal por decay exponencial")


class TrainingRequest(BaseModel):
    """
    Request para entrenar el modelo de fraude.

    La API extrae los datos automáticamente de la base de datos usando
    todos los datos históricos disponibles dentro del techo configurado,
    aplicando decay temporal exponencial y sampling balanceado.

    Decay formula: weight = exp(-λ * days_old)
    donde λ = ln(2) / half_life_days
    """

    # --- Rango temporal ---
    end_date: Optional[str] = Field(
        None,
        description=(
            "Fecha de referencia ('YYYY-MM-DD'). Si es None, usa REFERENCE_DATE del entorno "
            "o datetime.now(). Los pesos se calculan hacia atrás desde esta fecha."
        ),
        examples=["2026-03-03"],
    )
    start_date: Optional[str] = Field(
        None,
        description=(
            "Fecha inicial opcional ('YYYY-MM-DD'). Si es None, el sistema usa "
            "end_date - max_history_days como límite inferior automáticamente."
        ),
        examples=["2024-03-03"],
    )

    # --- Decay temporal ---
    half_life_days: int = Field(
        180,
        ge=30,
        le=1095,
        description=(
            "Vida media del decay en días. "
            "Ejemplo: 180 → los datos de hace 180 días valen el 50%. "
            "El sistema calcula λ = ln(2) / half_life_days automáticamente."
        ),
    )

    # --- Techo histórico (safety cap) ---
    max_history_days: int = Field(
        730,
        ge=90,
        le=1825,
        description=(
            "Máximo de días de historia a considerar (default 730 = 2 años). "
            "Evita OOM en producción. Solo aplica si start_date es None."
        ),
    )

    # --- Isolation Forest ---
    if_recent_months: int = Field(
        6,
        ge=1,
        le=24,
        description=(
            "Ventana de meses recientes para entrenar IsolationForest. "
            "IF no soporta sample_weight; limitarlo a datos recientes "
            "garantiza que detecte anomalías en el contexto actual."
        ),
    )

    # --- Hiperparámetros del pipeline ---
    optuna_trials: int = Field(30, ge=5, le=200, description="Número de trials de Optuna")
    undersampling_ratio: int = Field(
        4,
        ge=1,
        le=20,
        description="Ratio de legítimas por fraude (count-based). XGBoost aplica los pesos sobre este dataset.",
    )

    # --- Contexto de auditoría ---
    audit_id: Optional[int] = Field(None, description="ID del registro de auditoría creado por Java")
    triggered_by: str = Field(
        "manual",
        description="Quién disparó el entrenamiento: manual, scheduled, drift_detected",
    )


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
    # Información del decay para auditoría/trazabilidad
    half_life_days: int = Field(..., description="Vida media usada en el decay")
    effective_date_range: str = Field(..., description="Rango de fechas efectivo usado para extraer datos")