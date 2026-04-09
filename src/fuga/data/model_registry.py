# src/fuga/data/model_registry.py
"""
CRUD operations para el Model Registry de CHURN.
Maneja: churn_models, dataset_churn_prediction, self_training_audit_churn.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, List

from sqlalchemy.orm import Session
from sqlalchemy import desc

from fuga.data.db_models import ChurnModel, DatasetChurnPrediction, SelfTrainingAuditChurn

logger = logging.getLogger(__name__)


# ============================================================
# DATASET
# ============================================================

def save_dataset_info(
    session: Session,
    total_samples: int,
    count_train: int,
    count_test: int,
    churn_ratio: float,
    smote_applied: bool = True,
) -> int:
    dataset = DatasetChurnPrediction(
        total_samples=total_samples,
        count_train=count_train,
        count_test=count_test,
        churn_ratio=churn_ratio,
        smote_applied=smote_applied,
    )
    session.add(dataset)
    session.flush()
    logger.info(f"Dataset guardado: id={dataset.id_dataset}, samples={total_samples}")
    return dataset.id_dataset


# ============================================================
# MODEL
# ============================================================

def save_model_metadata(
    session: Session,
    model_version: str,
    algorithm: str,
    model_config: Dict,
    threshold: Optional[float] = None,
    dagshub_url: Optional[str] = None,
    file_path: Optional[str] = None,
    model_size_mb: Optional[float] = None,
    promotion_status: str = 'CHALLENGER',
    predecessor_model_id: Optional[int] = None,
) -> int:
    model = ChurnModel(
        model_version=model_version,
        algorithm=algorithm,
        model_config=model_config,
        threshold=threshold,
        dagshub_url=dagshub_url,
        file_path=file_path,
        model_size_mb=model_size_mb,
        promotion_status=promotion_status,
        predecessor_model_id=predecessor_model_id,
        is_active=False,
    )
    session.add(model)
    session.flush()
    logger.info(f"Modelo guardado: id={model.id_model}, version={model_version}, status={promotion_status}")
    return model.id_model


def get_current_champion(session: Session) -> Optional[ChurnModel]:
    champion = session.query(ChurnModel).filter(
        ChurnModel.is_active == True,
        ChurnModel.promotion_status == 'CHAMPION'
    ).first()
    if champion:
        logger.info(f"Champion actual: {champion.model_version} (id={champion.id_model})")
    else:
        logger.info("No hay champion activo en churn_models")
    return champion


def promote_model_to_champion(
    session: Session,
    model_id: int,
    promotion_reason: str,
) -> bool:
    new_champion = session.query(ChurnModel).filter(ChurnModel.id_model == model_id).first()
    if not new_champion:
        logger.error(f"Modelo {model_id} no encontrado")
        return False

    old_champion = get_current_champion(session)
    if old_champion:
        old_champion.is_active = False
        old_champion.promotion_status = 'RETIRED'
        logger.info(f"Champion anterior desactivado: {old_champion.model_version}")

    new_champion.is_active = True
    new_champion.promotion_status = 'CHAMPION'
    new_champion.promoted_at = datetime.utcnow()
    if old_champion:
        new_champion.predecessor_model_id = old_champion.id_model

    session.flush()
    logger.info(f"Nuevo CHAMPION promovido: {new_champion.model_version} — {promotion_reason}")
    return True


def update_model_dagshub_url(
    session: Session,
    model_id: int,
    dagshub_url: str,
    model_size_mb: float,
):
    model = session.query(ChurnModel).filter(ChurnModel.id_model == model_id).first()
    if model:
        model.dagshub_url = dagshub_url
        model.model_size_mb = model_size_mb
        session.flush()


# ============================================================
# AUDIT
# ============================================================

def save_complete_audit_record(
    session: Session,
    id_dataset: int,
    id_model: int,
    start_training: datetime,
    end_training: datetime,
    metrics: Dict,
    best_params: Dict,
    promotion_status: str,
    promotion_reason: Optional[str] = None,
    champion_metrics: Optional[Dict] = None,
    id_champion_model: Optional[int] = None,
    triggered_by: str = 'manual',
    trigger_details: Optional[Dict] = None,
    is_success: bool = True,
    error_message: Optional[str] = None,
) -> int:
    duration = int((end_training - start_training).total_seconds())

    audit = SelfTrainingAuditChurn(
        id_dataset=id_dataset,
        id_model=id_model,
        id_champion_model=id_champion_model,
        start_training=start_training,
        end_training=end_training,
        training_duration_seconds=duration,
        accuracy=metrics.get('accuracy'),
        precision_score=metrics.get('precision'),
        recall_score=metrics.get('recall'),
        f1_score=metrics.get('f1_score'),
        auc_roc=metrics.get('auc_roc'),
        champion_f1_score=champion_metrics.get('f1_score')  if champion_metrics else None,
        champion_recall=champion_metrics.get('recall')      if champion_metrics else None,
        champion_auc_roc=champion_metrics.get('auc_roc')    if champion_metrics else None,
        best_params=best_params,
        promotion_status=promotion_status,
        promotion_reason=promotion_reason,
        triggered_by=triggered_by,
        trigger_details=trigger_details,
        is_success=is_success,
        error_message=error_message,
    )
    session.add(audit)
    session.flush()
    logger.info(f"Audit guardado: id={audit.id_audit}, status={promotion_status}")
    return audit.id_audit


def get_champion_metrics_from_audit(session: Session, champion: ChurnModel) -> Optional[Dict]:
    """Lee las métricas del champion desde su último registro de auditoría."""
    audit = (
        session.query(SelfTrainingAuditChurn)
        .filter(SelfTrainingAuditChurn.id_model == champion.id_model)
        .order_by(desc(SelfTrainingAuditChurn.start_training))
        .first()
    )
    if not audit:
        return None
    return {
        "f1_score": float(audit.f1_score)    if audit.f1_score    else 0.0,
        "recall":   float(audit.recall_score) if audit.recall_score else 0.0,
        "auc_roc":  float(audit.auc_roc)      if audit.auc_roc      else 0.0,
    }
