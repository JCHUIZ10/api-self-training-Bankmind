# src/fraude/db_models.py
"""
SQLAlchemy models para el sistema de Model Registry & Audit.
Mapea las tablas: fraud_models, dataset_fraud_prediction, self_training_audit_fraud
"""

from sqlalchemy import (
    Column, BigInteger, String, Boolean, Numeric, Integer, 
    Text, TIMESTAMP, ForeignKey, CheckConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class FraudModel(Base):
    """Metadata de modelos de fraude. El archivo .pkl vive en DagsHub."""
    __tablename__ = 'fraud_models'
    
    id_model = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Identificación
    model_version = Column(String(50), nullable=False, unique=True)
    
    # Storage (DagsHub es primary storage)
    dagshub_url = Column(String(500))
    file_path = Column(String(255))
    model_size_mb = Column(Numeric(10, 2))
    
    # Architecture
    algorithm = Column(String(50), nullable=False)
    model_config = Column(JSONB)
    
    # Performance
    threshold = Column(Numeric(5, 4))
    
    # Status & Lifecycle
    promotion_status = Column(String(20), nullable=False, default='CHALLENGER')
    is_active = Column(Boolean, default=False)
    predecessor_model_id = Column(BigInteger, ForeignKey('fraud_models.id_model', ondelete='SET NULL'))
    
    # Timestamps
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    promoted_at = Column(TIMESTAMP)
    
    # Relationships
    predecessor = relationship('FraudModel', remote_side=[id_model], backref='successors')
    audit_records = relationship('SelfTrainingAuditFraud', foreign_keys='SelfTrainingAuditFraud.id_model', back_populates='model')
    champion_comparisons = relationship('SelfTrainingAuditFraud', foreign_keys='SelfTrainingAuditFraud.id_champion_model', back_populates='champion_model')
    
    # Indices
    __table_args__ = (
        Index('idx_fraud_models_is_active', 'is_active'),
        Index('idx_fraud_models_status', 'promotion_status'),
        Index('idx_fraud_models_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<FraudModel(id={self.id_model}, version='{self.model_version}', status='{self.promotion_status}')>"


class DatasetFraudPrediction(Base):
    """Información sobre datasets extraídos de operational_transactions"""
    __tablename__ = 'dataset_fraud_prediction'
    
    id_dataset = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Rango temporal
    start_date = Column(TIMESTAMP, nullable=False)
    end_date = Column(TIMESTAMP, nullable=False)
    
    # Tamaños
    total_samples = Column(Integer)
    count_train = Column(Integer)
    count_test = Column(Integer)
    
    # Balanceo
    fraud_ratio = Column(Numeric(5, 4))
    undersampling_ratio = Column(Integer)
    
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    audit_records = relationship('SelfTrainingAuditFraud', back_populates='dataset')
    
    # Constraints
    __table_args__ = (
        CheckConstraint('end_date >= start_date', name='chk_dataset_dates'),
        Index('idx_dataset_date_range', 'start_date', 'end_date'),
    )
    
    def __repr__(self):
        return f"<DatasetFraudPrediction(id={self.id_dataset}, {self.start_date} to {self.end_date})>"


class SelfTrainingAuditFraud(Base):
    """Auditoría completa de cada entrenamiento. Documenta métricas, decisiones y comparaciones."""
    __tablename__ = 'self_training_audit_fraud'
    
    id_audit = Column(BigInteger, primary_key=True, autoincrement=True)
    
    # Relaciones
    id_dataset = Column(BigInteger, ForeignKey('dataset_fraud_prediction.id_dataset', ondelete='RESTRICT'), nullable=False)
    id_model = Column(BigInteger, ForeignKey('fraud_models.id_model', ondelete='RESTRICT'), nullable=False)
    id_champion_model = Column(BigInteger, ForeignKey('fraud_models.id_model', ondelete='SET NULL'))
    
    # Timing
    start_training = Column(TIMESTAMP)
    end_training = Column(TIMESTAMP)
    training_duration_seconds = Column(Integer)
    
    # Métricas del CHALLENGER (modelo nuevo)
    accuracy = Column(Numeric(6, 5))
    precision_score = Column(Numeric(6, 5))
    recall_score = Column(Numeric(6, 5))
    f1_score = Column(Numeric(6, 5))
    auc_roc = Column(Numeric(6, 5))
    optimal_threshold = Column(Numeric(5, 4))
    
    # Métricas del CHAMPION (para comparación)
    champion_f1_score = Column(Numeric(6, 5))
    champion_recall = Column(Numeric(6, 5))
    champion_auc_roc = Column(Numeric(6, 5))
    
    # Optuna
    optuna_best_f1 = Column(Numeric(6, 5))
    optuna_best_params = Column(JSONB)
    
    # Decision
    promotion_status = Column(String(20), nullable=False)
    promotion_reason = Column(Text)
    
    # Trigger
    triggered_by = Column(String(50))
    trigger_details = Column(JSONB)
    
    # Error handling
    is_success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    # Relationships
    dataset = relationship('DatasetFraudPrediction', back_populates='audit_records')
    model = relationship('FraudModel', foreign_keys=[id_model], back_populates='audit_records')
    champion_model = relationship('FraudModel', foreign_keys=[id_champion_model], back_populates='champion_comparisons')
    
    # Constraints & Indices
    __table_args__ = (
        CheckConstraint(
            'end_training IS NULL OR start_training IS NULL OR end_training >= start_training',
            name='chk_training_dates'
        ),
        Index('idx_audit_model', 'id_model'),
        Index('idx_audit_dataset', 'id_dataset'),
        Index('idx_audit_champion', 'id_champion_model'),
        Index('idx_audit_success', 'is_success'),
        Index('idx_audit_created', 'start_training'),
    )
    
    def __repr__(self):
        return f"<SelfTrainingAuditFraud(id={self.id_audit}, model_id={self.id_model}, status='{self.promotion_status}')>"
