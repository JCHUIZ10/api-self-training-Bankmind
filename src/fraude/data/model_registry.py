# src/fraude/model_registry.py
"""
CRUD operations para el sistema de Model Registry & Audit.
Maneja la interacción con fraud_models, dataset_fraud_prediction y self_training_audit_fraud.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc

from fraude.data.db_models import FraudModel, DatasetFraudPrediction, SelfTrainingAuditFraud

logger = logging.getLogger(__name__)


# ========================================
# DATASET OPERATIONS
# ========================================

def save_dataset_info(
    session: Session,
    start_date: str,
    end_date: str,
    total_samples: int,
    count_train: int,
    count_test: int,
    fraud_ratio: float,
    undersampling_ratio: int
) -> int:
    """
    Guarda información del dataset usado para entrenamiento.
    
    Returns:
        id_dataset: ID del registro creado
    """
    dataset = DatasetFraudPrediction(
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date),
        total_samples=total_samples,
        count_train=count_train,
        count_test=count_test,
        fraud_ratio=fraud_ratio,
        undersampling_ratio=undersampling_ratio
    )
    
    session.add(dataset)
    session.flush()  # Para obtener el id_dataset
    
    logger.info(f"✅ Dataset guardado: id={dataset.id_dataset}, samples={total_samples}")
    return dataset.id_dataset


# ========================================
# MODEL OPERATIONS
# ========================================

def save_model_metadata(
    session: Session,
    model_version: str,
    algorithm: str,
    model_config: Dict,
    threshold: float,
    dagshub_url: Optional[str] = None,
    file_path: Optional[str] = None,
    model_size_mb: Optional[float] = None,
    promotion_status: str = 'CHALLENGER',
    predecessor_model_id: Optional[int] = None
) -> int:
    """
    Guarda metadata de un modelo entrenado.
    
    Args:
        model_version: Versión única del modelo (ej. "fraud_v1.0_20260216")
        algorithm: Nombre del algoritmo (ej. "XGBoost + IsolationForest")
        model_config: Dict completo de configuración
        threshold: Optimal threshold
        dagshub_url: URL en DagsHub (si ya se subió)
        promotion_status: CHAMPION, CHALLENGER, REJECTED
        
    Returns:
        id_model: ID del registro creado
    """
    model = FraudModel(
        model_version=model_version,
        algorithm=algorithm,
        model_config=model_config,
        threshold=threshold,
        dagshub_url=dagshub_url,
        file_path=file_path,
        model_size_mb=model_size_mb,
        promotion_status=promotion_status,
        predecessor_model_id=predecessor_model_id,
        is_active=False  # Solo se activa después de comparación
    )
    
    session.add(model)
    session.flush()
    
    logger.info(f"✅ Modelo guardado: id={model.id_model}, version={model_version}, status={promotion_status}")
    return model.id_model


def get_current_champion(session: Session) -> Optional[FraudModel]:
    """
    Obtiene el modelo CHAMPION actualmente en producción.
    
    Returns:
        FraudModel o None si no hay champion
    """
    champion = session.query(FraudModel).filter(
        FraudModel.is_active == True,
        FraudModel.promotion_status == 'CHAMPION'
    ).first()
    
    if champion:
        logger.info(f"🏆 Champion actual: {champion.model_version} (id={champion.id_model})")
    else:
        logger.info("⚠️ No hay champion activo en este momento")
    
    return champion


def promote_model_to_champion(
    session: Session,
    model_id: int,
    promotion_reason: str
) -> bool:
    """
    Promueve un modelo a CHAMPION y desactiva el anterior.
    
    Args:
        model_id: ID del modelo a promover
        promotion_reason: Razón de la promoción
        
    Returns:
        True si se promovió exitosamente
    """
    # Obtener el modelo a promover
    new_champion = session.query(FraudModel).filter(FraudModel.id_model == model_id).first()
    
    if not new_champion:
        logger.error(f"❌ Modelo {model_id} no encontrado")
        return False
    
    # Desactivar champion anterior (si existe)
    old_champion = get_current_champion(session)
    if old_champion:
        old_champion.is_active = False
        old_champion.promotion_status = 'RETIRED'
        logger.info(f"♻️ Champion anterior desactivado: {old_champion.model_version}")
    
    # Activar nuevo champion
    new_champion.is_active = True
    new_champion.promotion_status = 'CHAMPION'
    new_champion.promoted_at = datetime.utcnow()
    if old_champion:
        new_champion.predecessor_model_id = old_champion.id_model
    
    session.flush()
    
    logger.info(f"🏆 Nuevo CHAMPION promovido: {new_champion.model_version}")
    logger.info(f"   Razón: {promotion_reason}")
    
    return True


def update_model_dagshub_url(session: Session, model_id: int, dagshub_url: str, model_size_mb: float):
    """Actualiza la URL de DagsHub después de subir el modelo"""
    model = session.query(FraudModel).filter(FraudModel.id_model == model_id).first()
    if model:
        model.dagshub_url = dagshub_url
        model.model_size_mb = model_size_mb
        session.flush()
        logger.info(f"✅ DagsHub URL actualizada para modelo {model_id}")


# ========================================
# AUDIT OPERATIONS
# ========================================

def create_audit_record(
    session: Session,
    triggered_by: str = 'manual',
    trigger_details: Optional[Dict] = None
) -> int:
    """
    Crea un registro de auditoría inicial (usado por Java al disparar el entrenamiento).
    Python luego lo actualiza con los resultados.
    
    Args:
        triggered_by: "manual", "scheduled", "drift_detected"
        trigger_details: Info adicional del trigger (ej. {"psi": 0.28})
        
    Returns:
        id_audit: ID del registro creado
    """
    audit = SelfTrainingAuditFraud(
        id_dataset=0,  # Temporal, se actualiza después
        id_model=0,    # Temporal, se actualiza después
        start_training=datetime.utcnow(),
        triggered_by=triggered_by,
        trigger_details=trigger_details,
        is_success=False,  # Pesimista hasta que Python termine
        promotion_status='PENDING'
    )
    
    session.add(audit)
    session.flush()
    
    logger.info(f"📝 Audit record inicial creado: id={audit.id_audit}, triggered_by={triggered_by}")
    return audit.id_audit


def update_audit_with_results(
    session: Session,
    audit_id: int,
    id_dataset: int,
    id_model: int,
    end_training: datetime,
    metrics: Dict,
    optuna_result: Dict,
    promotion_status: str,
    promotion_reason: Optional[str] = None,
    champion_metrics: Optional[Dict] = None,
    id_champion_model: Optional[int] = None,
    is_success: bool = True,
    error_message: Optional[str] = None
) -> bool:
    """
    Actualiza un registro de auditoría con los resultados del entrenamiento.
    Llamado por Python después de completar el entrenamiento.
    
    Args:
        audit_id: ID del audit record a actualizar
        id_dataset: ID del dataset usado
        id_model: ID del modelo generado
        end_training: Timestamp de fin
        metrics: Dict con métricas del modelo
        optuna_result: Dict con resultados de Optuna
        promotion_status: PROMOTED, REJECTED, PENDING
        champion_metrics: Dict con métricas del champion (opcional)
        id_champion_model: ID del champion comparado (opcional)
        is_success: Si el entrenamiento terminó exitosamente
        error_message: Mensaje de error si falló
        
    Returns:
        True si se actualizó correctamente
    """
    audit = session.query(SelfTrainingAuditFraud).filter(
        SelfTrainingAuditFraud.id_audit == audit_id
    ).first()
    
    if not audit:
        logger.error(f"❌ Audit record {audit_id} no encontrado")
        return False
    
    # Actualizar relaciones
    audit.id_dataset = id_dataset
    audit.id_model = id_model
    audit.id_champion_model = id_champion_model
    
    # Actualizar timing
    audit.end_training = end_training
    if audit.start_training:
        audit.training_duration_seconds = int((end_training - audit.start_training).total_seconds())
    
    # Actualizar métricas del challenger
    audit.accuracy = metrics.get('accuracy')
    audit.precision_score = metrics.get('precision')
    audit.recall_score = metrics.get('recall')
    audit.f1_score = metrics.get('f1_score')
    audit.auc_roc = metrics.get('auc_roc')
    audit.optimal_threshold = metrics.get('optimal_threshold')
    
    # Actualizar métricas del champion (si hubo comparación)
    if champion_metrics:
        audit.champion_f1_score = champion_metrics.get('f1_score')
        audit.champion_recall = champion_metrics.get('recall')
        audit.champion_auc_roc = champion_metrics.get('auc_roc')
    
    # Actualizar Optuna
    audit.optuna_best_f1 = optuna_result.get('best_f1_score')
    audit.optuna_best_params = optuna_result.get('best_params')
    
    # Actualizar decisión
    audit.promotion_status = promotion_status
    audit.promotion_reason = promotion_reason
    
    # Actualizar estado
    audit.is_success = is_success
    audit.error_message = error_message
    
    session.flush()
    
    logger.info(f"✅ Audit record actualizado: id={audit_id}, status={promotion_status}, success={is_success}")
    return True


def save_complete_audit_record(
    session: Session,
    id_dataset: int,
    id_model: int,
    start_training: datetime,
    end_training: datetime,
    metrics: Dict,
    optuna_result: Dict,
    promotion_status: str,
    promotion_reason: Optional[str] = None,
    champion_metrics: Optional[Dict] = None,
    id_champion_model: Optional[int] = None,
    triggered_by: str = 'manual',
    trigger_details: Optional[Dict] = None,
    is_success: bool = True,
    error_message: Optional[str] = None
) -> int:
    """
    Crea un registro de auditoría completo de una vez (para flujo manual).
    Evita el problema de foreign keys con valores temporales.
    
    Args:
        id_dataset: ID del dataset (debe existir)
        id_model: ID del modelo (debe existir)
        start_training: Timestamp de inicio
        end_training: Timestamp de fin
        metrics: Dict con métricas del modelo
        optuna_result: Dict con resultados de Optuna
        promotion_status: PROMOTED, REJECTED, PENDING
        champion_metrics: Dict con métricas del champion (opcional)
        id_champion_model: ID del champion comparado (opcional)
        triggered_by: "manual", "scheduled", "drift_detected"
        trigger_details: Info adicional del trigger
        is_success: Si el entrenamiento terminó exitosamente
        error_message: Mensaje de error si falló
        
    Returns:
        id_audit: ID del registro creado
    """
    training_duration = int((end_training - start_training).total_seconds())
    
    audit = SelfTrainingAuditFraud(
        id_dataset=id_dataset,
        id_model=id_model,
        id_champion_model=id_champion_model,
        
        start_training=start_training,
        end_training=end_training,
        training_duration_seconds=training_duration,
        
        # Métricas del challenger
        accuracy=metrics.get('accuracy'),
        precision_score=metrics.get('precision'),
        recall_score=metrics.get('recall'),
        f1_score=metrics.get('f1_score'),
        auc_roc=metrics.get('auc_roc'),
        optimal_threshold=metrics.get('optimal_threshold'),
        
        # Métricas del champion (si hubo comparación)
        champion_f1_score=champion_metrics.get('f1_score') if champion_metrics else None,
        champion_recall=champion_metrics.get('recall') if champion_metrics else None,
        champion_auc_roc=champion_metrics.get('auc_roc') if champion_metrics else None,
        
        # Optuna
        optuna_best_f1=optuna_result.get('best_f1_score'),
        optuna_best_params=optuna_result.get('best_params'),
        
        # Decision
        promotion_status=promotion_status,
        promotion_reason=promotion_reason,
        
        # Trigger
        triggered_by=triggered_by,
        trigger_details=trigger_details,
        
        # Error handling
        is_success=is_success,
        error_message=error_message
    )
    
    session.add(audit)
    session.flush()
    
    logger.info(f"📝 Auditoría completa guardada: id={audit.id_audit}, status={promotion_status}")
    return audit.id_audit


def get_recent_audits(session: Session, limit: int = 10) -> List[SelfTrainingAuditFraud]:
    """Obtiene los últimos N registros de auditoría"""
    return session.query(SelfTrainingAuditFraud).order_by(
        desc(SelfTrainingAuditFraud.start_training)
    ).limit(limit).all()
