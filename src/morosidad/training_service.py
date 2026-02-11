# src/morosidad/training_service.py
import logging
import time
import base64
import io
import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap

from morosidad.morosidad_schema import (
    TrainingRequest, TrainingResponse,
    TrainingMetrics, OptunaResult
)

logger = logging.getLogger(__name__)

# Orden exacto de las columnas que espera el modelo
FEATURE_COLUMNS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'UTILIZATION_RATE'
]


def calcular_ks_statistic(y_true, y_proba):
    """Calcula la estadística KS (Kolmogorov-Smirnov)."""
    from scipy.stats import ks_2samp
    pos_proba = y_proba[y_true == 1]
    neg_proba = y_proba[y_true == 0]
    if len(pos_proba) == 0 or len(neg_proba) == 0:
        return 0.0
    ks_stat, _ = ks_2samp(pos_proba, neg_proba)
    return float(ks_stat)


def entrenar_modelo(request: TrainingRequest) -> TrainingResponse:
    """
    Pipeline completo de entrenamiento:
    1. Preparar datos
    2. Optimizar con Optuna
    3. Entrenar ensemble final
    4. Calcular métricas
    5. Serializar modelo
    """
    start_time = time.time()
    logger.info(f"🚀 Iniciando entrenamiento con {len(request.samples)} muestras")

    # =========================================
    # 1. PREPARAR DATOS
    # =========================================
    samples_dicts = [s.model_dump() for s in request.samples]
    df = pd.DataFrame(samples_dicts)

    X = df[FEATURE_COLUMNS].copy()
    y = df['default_payment_next_month'].copy()
    weights = df['sample_weight'].values

    # Distribución de clases
    class_dist = y.value_counts().to_dict()
    class_dist = {str(k): int(v) for k, v in class_dist.items()}
    logger.info(f"📊 Distribución de clases: {class_dist}")

    # Scale pos weight dinámico
    n_negative = int((y == 0).sum())
    n_positive = int((y == 1).sum())
    scale_pos_weight = n_negative / max(n_positive, 1)
    logger.info(f"⚖️ scale_pos_weight calculado: {scale_pos_weight:.4f}")

    # Split estratificado
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"📂 Train: {len(X_train)} | Test: {len(X_test)}")

    # =========================================
    # 2. OPTIMIZAR CON OPTUNA
    # =========================================
    logger.info(f"🔍 Iniciando optimización Optuna ({request.optuna_trials} trials)...")

    # Silenciar los logs de Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """Función objetivo para Optuna: optimiza AUC-ROC del ensemble."""

        # Hiperparámetros XGBoost
        xgb_params = {
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 800),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42,
            'verbosity': 0
        }

        # Hiperparámetros LightGBM
        lgbm_params = {
            'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 60),
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 800),
            'scale_pos_weight': scale_pos_weight,
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'verbose': -1
        }

        # Hiperparámetros Random Forest
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 15),
            'class_weight': 'balanced',
            'random_state': 42
        }

        # Pesos del ensemble
        w_xgb = trial.suggest_int('ensemble_weight_xgb', 1, 5)
        w_lgbm = trial.suggest_int('ensemble_weight_lgbm', 1, 3)
        w_rf = trial.suggest_int('ensemble_weight_rf', 1, 2)

        # Crear ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgboost', xgb.XGBClassifier(**xgb_params)),
                ('lightgbm', lgb.LGBMClassifier(**lgbm_params)),
                ('rf', RandomForestClassifier(**rf_params))
            ],
            voting='soft',
            weights=[w_xgb, w_lgbm, w_rf]
        )

        ensemble.fit(X_train, y_train, sample_weight=w_train)
        y_proba = ensemble.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba, sample_weight=w_test)
        return auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=request.optuna_trials)

    best_trial = study.best_trial
    best_params = best_trial.params
    logger.info(f"✅ Mejor trial #{best_trial.number}: AUC = {best_trial.value:.4f}")

    # =========================================
    # 3. ENTRENAR MODELO FINAL CON MEJORES PARÁMETROS
    # =========================================
    logger.info("🏗️ Entrenando modelo final con los mejores parámetros...")

    xgb_final = xgb.XGBClassifier(
        learning_rate=best_params['xgb_learning_rate'],
        max_depth=best_params['xgb_max_depth'],
        n_estimators=best_params['xgb_n_estimators'],
        subsample=best_params['xgb_subsample'],
        colsample_bytree=best_params['xgb_colsample_bytree'],
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        verbosity=0
    )

    lgbm_final = lgb.LGBMClassifier(
        learning_rate=best_params['lgbm_learning_rate'],
        num_leaves=best_params['lgbm_num_leaves'],
        n_estimators=best_params['lgbm_n_estimators'],
        scale_pos_weight=scale_pos_weight,
        objective='binary',
        metric='auc',
        random_state=42,
        verbose=-1
    )

    rf_final = RandomForestClassifier(
        n_estimators=best_params['rf_n_estimators'],
        max_depth=best_params['rf_max_depth'],
        class_weight='balanced',
        random_state=42
    )

    w_xgb = best_params['ensemble_weight_xgb']
    w_lgbm = best_params['ensemble_weight_lgbm']
    w_rf = best_params['ensemble_weight_rf']

    final_ensemble = VotingClassifier(
        estimators=[
            ('xgboost', xgb_final),
            ('lightgbm', lgbm_final),
            ('rf', rf_final)
        ],
        voting='soft',
        weights=[w_xgb, w_lgbm, w_rf]
    )

    final_ensemble.fit(X_train, y_train, sample_weight=w_train)

    # =========================================
    # 4. CALCULAR MÉTRICAS
    # =========================================
    logger.info("📏 Calculando métricas sobre el test set...")

    y_proba = final_ensemble.predict_proba(X_test)[:, 1]
    y_pred = final_ensemble.predict(X_test)

    auc_roc = roc_auc_score(y_test, y_proba, sample_weight=w_test)
    ks_stat = calcular_ks_statistic(y_test.values, y_proba)
    gini = 2 * auc_roc - 1

    training_time = time.time() - start_time

    metrics = TrainingMetrics(
        auc_roc=round(auc_roc, 4),
        ks_statistic=round(ks_stat, 4),
        gini_coefficient=round(gini, 4),
        accuracy=round(accuracy_score(y_test, y_pred, sample_weight=w_test), 4),
        precision=round(precision_score(y_test, y_pred, sample_weight=w_test, zero_division=0), 4),
        recall=round(recall_score(y_test, y_pred, sample_weight=w_test, zero_division=0), 4),
        f1_score=round(f1_score(y_test, y_pred, sample_weight=w_test, zero_division=0), 4),
        training_time_sec=round(training_time, 2)
    )

    logger.info(f"📊 Métricas: AUC={metrics.auc_roc} | KS={metrics.ks_statistic} | "
                f"Gini={metrics.gini_coefficient} | F1={metrics.f1_score}")

    # =========================================
    # 5. CREAR SHAP EXPLAINER
    # =========================================
    logger.info("🔬 Creando SHAP explainer...")
    xgb_model = final_ensemble.named_estimators_['xgboost']
    try:
        explainer = shap.TreeExplainer(xgb_model)
        logger.info("✅ SHAP explainer creado correctamente")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo crear SHAP explainer: {e}")
        explainer = None

    # =========================================
    # 6. SERIALIZAR MODELO
    # =========================================
    logger.info("📦 Serializando modelo...")

    model_package = {
        'modelo_prediccion': final_ensemble,
        'shap_explainer': explainer,
        'meta_info': {
            'version': 'auto-trained',
            'features': FEATURE_COLUMNS,
            'scale_pos_weight': scale_pos_weight,
            'training_samples': len(X_train),
            'auc_roc': metrics.auc_roc
        }
    }

    buffer = io.BytesIO()
    joblib.dump(model_package, buffer)
    model_bytes = base64.b64encode(buffer.getvalue()).decode('utf-8')
    logger.info(f"📦 Modelo serializado: {len(model_bytes)} caracteres base64")

    # =========================================
    # 7. CONSTRUIR RESPONSE
    # =========================================
    assembly_config = {
        "architecture": "VotingClassifier",
        "voting_strategy": "soft",
        "weights_assigned": [w_xgb, w_lgbm, w_rf],
        "order_estimators": ["xgboost", "lightgbm", "rf"],
        "random_seed": 42,
        "features_input": FEATURE_COLUMNS,
        "internal_components": {
            "xgboost": {
                "learning_rate": best_params['xgb_learning_rate'],
                "max_depth": best_params['xgb_max_depth'],
                "n_estimators": best_params['xgb_n_estimators'],
                "subsample": best_params['xgb_subsample'],
                "colsample_bytree": best_params['xgb_colsample_bytree'],
                "scale_pos_weight": scale_pos_weight
            },
            "lightgbm": {
                "learning_rate": best_params['lgbm_learning_rate'],
                "num_leaves": best_params['lgbm_num_leaves'],
                "n_estimators": best_params['lgbm_n_estimators'],
                "scale_pos_weight": scale_pos_weight
            },
            "rf": {
                "n_estimators": best_params['rf_n_estimators'],
                "max_depth": best_params['rf_max_depth'],
                "class_weight": "balanced"
            }
        }
    }

    optuna_result = OptunaResult(
        trial_number=best_trial.number,
        objective_value=round(best_trial.value, 4),
        metric_optimized="auc_roc",
        best_params=best_params
    )

    response = TrainingResponse(
        metrics=metrics,
        optuna_result=optuna_result,
        model_base64=model_bytes,
        assembly_config=assembly_config,
        total_samples=len(df),
        train_samples=len(X_train),
        test_samples=len(X_test),
        class_distribution=class_dist,
        scale_pos_weight=round(scale_pos_weight, 4)
    )

    logger.info(f"✅ Entrenamiento completado en {training_time:.1f}s")
    return response
