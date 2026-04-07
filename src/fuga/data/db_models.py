# src/fuga/data/db_models.py
"""
SQLAlchemy models para el sistema de Model Registry & Audit de CHURN.
Tablas: churn_models, dataset_churn_prediction, self_training_audit_churn
"""

from datetime import datetime
from sqlalchemy import (
    Column, BigInteger, String, Boolean, Numeric, Integer,
    Text, TIMESTAMP, ForeignKey, CheckConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ChurnModel(Base):
    """Metadata de modelos de churn. El archivo .pkl vive en DagsHub."""
    __tablename__ = 'churn_models'

    id_model       = Column(BigInteger, primary_key=True, autoincrement=True)
    model_version  = Column(String(50),  nullable=False, unique=True)
    dagshub_url    = Column(String(500))
    file_path      = Column(String(255))
    model_size_mb  = Column(Numeric(10, 2))
    algorithm      = Column(String(50),  nullable=False)
    model_config   = Column(JSONB)
    threshold      = Column(Numeric(5, 4))

    # CHAMPION | CHALLENGER | REJECTED | RETIRED
    promotion_status    = Column(String(20), nullable=False, default='CHALLENGER')
    is_active           = Column(Boolean, default=False)
    predecessor_model_id = Column(
        BigInteger, ForeignKey('churn_models.id_model', ondelete='SET NULL')
    )

    created_at   = Column(TIMESTAMP, default=datetime.utcnow)
    promoted_at  = Column(TIMESTAMP)

    predecessor = relationship('ChurnModel', remote_side=[id_model], backref='successors')
    audit_records = relationship(
        'SelfTrainingAuditChurn',
        foreign_keys='SelfTrainingAuditChurn.id_model',
        back_populates='model'
    )
    champion_comparisons = relationship(
        'SelfTrainingAuditChurn',
        foreign_keys='SelfTrainingAuditChurn.id_champion_model',
        back_populates='champion_model'
    )

    __table_args__ = (
        Index('idx_churn_models_is_active',  'is_active'),
        Index('idx_churn_models_status',     'promotion_status'),
        Index('idx_churn_models_created',    'created_at'),
    )

    def __repr__(self):
        return f"<ChurnModel(id={self.id_model}, version='{self.model_version}', status='{self.promotion_status}')>"


class DatasetChurnPrediction(Base):
    """Información sobre datasets de entrenamiento extraídos de account_details."""
    __tablename__ = 'dataset_churn_prediction'

    id_dataset     = Column(BigInteger, primary_key=True, autoincrement=True)
    total_samples  = Column(Integer)
    count_train    = Column(Integer)
    count_test     = Column(Integer)
    churn_ratio    = Column(Numeric(5, 4))   # ratio de churn en el dataset
    smote_applied  = Column(Boolean, default=True)
    created_at     = Column(TIMESTAMP, default=datetime.utcnow)

    audit_records = relationship('SelfTrainingAuditChurn', back_populates='dataset')

    __table_args__ = (
        Index('idx_dataset_churn_created', 'created_at'),
    )

    def __repr__(self):
        return f"<DatasetChurnPrediction(id={self.id_dataset}, samples={self.total_samples})>"


class SelfTrainingAuditChurn(Base):
    """Auditoría completa de cada ciclo de entrenamiento de churn."""
    __tablename__ = 'self_training_audit_churn'

    id_audit = Column(BigInteger, primary_key=True, autoincrement=True)

    # Relaciones
    id_dataset       = Column(BigInteger, ForeignKey('dataset_churn_prediction.id_dataset', ondelete='RESTRICT'), nullable=False)
    id_model         = Column(BigInteger, ForeignKey('churn_models.id_model',              ondelete='RESTRICT'), nullable=False)
    id_champion_model = Column(BigInteger, ForeignKey('churn_models.id_model',             ondelete='SET NULL'))

    # Timing
    start_training           = Column(TIMESTAMP)
    end_training             = Column(TIMESTAMP)
    training_duration_seconds = Column(Integer)

    # Métricas del CHALLENGER
    accuracy        = Column(Numeric(6, 5))
    precision_score = Column(Numeric(6, 5))
    recall_score    = Column(Numeric(6, 5))
    f1_score        = Column(Numeric(6, 5))
    auc_roc         = Column(Numeric(6, 5))

    # Métricas del CHAMPION (para comparación)
    champion_f1_score = Column(Numeric(6, 5))
    champion_recall   = Column(Numeric(6, 5))
    champion_auc_roc  = Column(Numeric(6, 5))

    # Hiperparámetros del GridSearch
    best_params = Column(JSONB)

    # Decisión
    promotion_status = Column(String(20), nullable=False)
    promotion_reason = Column(Text)

    # Trigger
    triggered_by    = Column(String(50))
    trigger_details = Column(JSONB)

    # Estado
    is_success    = Column(Boolean, default=True)
    error_message = Column(Text)

    # Relaciones ORM
    dataset = relationship('DatasetChurnPrediction', back_populates='audit_records')
    model   = relationship('ChurnModel', foreign_keys=[id_model],          back_populates='audit_records')
    champion_model = relationship('ChurnModel', foreign_keys=[id_champion_model], back_populates='champion_comparisons')

    __table_args__ = (
        CheckConstraint(
            'end_training IS NULL OR start_training IS NULL OR end_training >= start_training',
            name='chk_churn_training_dates'
        ),
        Index('idx_audit_churn_model',    'id_model'),
        Index('idx_audit_churn_dataset',  'id_dataset'),
        Index('idx_audit_churn_success',  'is_success'),
        Index('idx_audit_churn_created',  'start_training'),
    )

    def __repr__(self):
        return f"<SelfTrainingAuditChurn(id={self.id_audit}, model={self.id_model}, status='{self.promotion_status}')>"
