# src/fuga/core/training/training_pipeline.py
"""
Pipeline de auto-entrenamiento del modelo de CHURN.

Flujo:
1.  Extrae datos de la BD (account_details + joins)
2.  Ingeniería de features (TenureByAge, BalanceSalaryRatio, CreditScoreGivenAge)
3.  Encoding de variables categóricas (Geography → dummies, Gender → binario)
4.  Split train/test estratificado (80/20)
5.  Escala con StandardScaler
6.  Balancea con SMOTE sobre el conjunto de entrenamiento
7.  Optimiza hiperparámetros con Optuna (XGBoost, scoring=roc_auc)
8.  Evalúa el challenger en el conjunto de test
9.  Champion/Challenger: compara con el modelo en producción (churn_models tabla)
10. Si el challenger gana: sube combo-pack a DagsHub + hot-reload al servidor principal
11. Persiste dataset, modelo y auditoría en la BD
12. Devuelve TrainingResponse
"""

import io
import logging
import time
from datetime import datetime
from typing import Optional, Tuple, Dict

import joblib
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fuga.data.data_extraction import extract_training_data
from fuga.data.db_config import get_db_session
from fuga.data import model_registry
from fuga.infrastructure.dagshub import (
    init_dagshub_connection,
    upload_champion,
    verify_champion_integrity,
    notify_hot_reload,
)
from fuga.schemas.churn import TrainingRequest, TrainingResponse, TrainingMetrics

logger = logging.getLogger(__name__)

# Score mínimo de mejora para promover el challenger (evita ruido estadístico)
_MIN_IMPROVEMENT = 0.005

FEATURE_NAMES = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'TenureByAge', 'BalanceSalaryRatio', 'CreditScoreGivenAge',
    'Geography_Germany', 'Geography_Spain',
]


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Aplica feature engineering, encoding y devuelve (X, y)."""
    df = df.copy()
    eps = 1e-9

    # Feature engineering
    df['TenureByAge']         = df['Tenure']      / (df['Age']               + eps)
    df['BalanceSalaryRatio']  = df['Balance']      / (df['EstimatedSalary']   + eps)
    df['CreditScoreGivenAge'] = df['CreditScore']  / (df['Age']               + eps)

    # Gender → binario (robusto a 'Male'/'Hombre'/etc.)
    df['Gender'] = df['Gender'].str.strip().str.lower().map({
        'male': 1, 'female': 0,
        'hombre': 1, 'mujer': 0,
    }).fillna(0).astype(int)

    # Geography → dummies (drop_first implícito: France queda excluida)
    geo_dummies = pd.get_dummies(df['Geography'], prefix='Geography')
    for col in ('Geography_Germany', 'Geography_Spain'):
        if col not in geo_dummies.columns:
            geo_dummies[col] = 0
    df = pd.concat([df, geo_dummies[['Geography_Germany', 'Geography_Spain']]], axis=1)
    df.drop(columns=['Geography'], inplace=True)

    y = df['Exited'].astype(int)
    X = df[FEATURE_NAMES]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# CHAMPION / CHALLENGER
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_promotion(
    new_metrics: Dict,
    champion_metrics: Optional[Dict],
) -> Tuple[bool, str]:
    """
    Score compuesto = AUC·40% + F1·30% + Recall·30%.
    Umbral mínimo: _MIN_IMPROVEMENT (0.5 pp).
    """
    if champion_metrics is None:
        return True, "Primer modelo registrado — promovido automaticamente."

    new_auc   = new_metrics.get("auc_roc",  0.0)
    champ_auc = champion_metrics.get("auc_roc", 0.0)
    new_f1    = new_metrics.get("f1_score",  0.0)
    champ_f1  = champion_metrics.get("f1_score", 0.0)
    new_rec   = new_metrics.get("recall",    0.0)
    champ_rec = champion_metrics.get("recall", 0.0)

    # Detectar AUC no confiable (bug histórico)
    if champ_auc < 0.01 and champ_f1 > 0.3:
        new_score   = 0.5 * new_f1   + 0.5 * new_rec
        champ_score = 0.5 * champ_f1 + 0.5 * champ_rec
        label = "F1+Recall (AUC campeon no confiable)"
    else:
        new_score   = 0.4 * new_auc   + 0.3 * new_f1   + 0.3 * new_rec
        champ_score = 0.4 * champ_auc + 0.3 * champ_f1 + 0.3 * champ_rec
        label = "Score Compuesto (AUC*40 + F1*30 + Recall*30)"

    delta = new_score - champ_score
    logger.info(
        "[Churn C/C] %s: challenger=%.4f vs campeon=%.4f (delta=%+.4f, min=%.4f)",
        label, new_score, champ_score, delta, _MIN_IMPROVEMENT,
    )

    if delta >= _MIN_IMPROVEMENT:
        return True, f"{label}: {champ_score:.4f} -> {new_score:.4f} (+{delta:.4f})."
    if delta > -_MIN_IMPROVEMENT:
        return False, f"Sin mejora significativa ({label}: delta={delta:+.4f}). Campeon se mantiene."
    return False, f"{label} empeoro: {champ_score:.4f} -> {new_score:.4f} ({delta:+.4f}). Campeon se mantiene."


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_modelo(request: TrainingRequest) -> TrainingResponse:
    """
    Punto de entrada del pipeline de auto-entrenamiento de churn.
    Se ejecuta en un thread pool desde el router (no bloquea el event loop).
    """
    start_time = time.time()
    logger.info("=== INICIO AUTO-ENTRENAMIENTO CHURN (self-training API) ===")

    # 1. Extracción
    df_raw = extract_training_data()
    total_samples = len(df_raw)
    churn_ratio = float(df_raw['Exited'].mean())
    logger.info(f"Dataset: {total_samples} registros, churn={churn_ratio:.2%}")

    # 2. Preprocesamiento
    X, y = _preprocess(df_raw)

    # 3. Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Escalado
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 5. SMOTE
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    logger.info(f"Balance pre-SMOTE: {neg} no-churn / {pos} churn")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_s, y_train)
    logger.info(f"Post-SMOTE: {len(X_res)} muestras, {int(y_res.sum())} churn")

    # 6. Optuna — búsqueda de hiperparámetros XGBoost
    n_trials = request.optuna_trials
    logger.info(f"[Optuna] Iniciando optimización de hiperparámetros ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators':      trial.suggest_int('n_estimators',      100, 400),
            'max_depth':         trial.suggest_int('max_depth',          3,   8),
            'learning_rate':     trial.suggest_float('learning_rate',    0.01, 0.3,  log=True),
            'subsample':         trial.suggest_float('subsample',        0.6,  1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6,  1.0),
            'min_child_weight':  trial.suggest_int('min_child_weight',   1,   10),
            'reg_alpha':         trial.suggest_float('reg_alpha',        0.0,  5.0),
            'reg_lambda':        trial.suggest_float('reg_lambda',       0.0,  5.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': 1,
            'verbosity': 0,
        }
        clf = XGBClassifier(**params)
        clf.fit(X_res, y_res)
        proba = clf.predict_proba(X_test_s)[:, 1]
        return float(roc_auc_score(y_test, proba))

    study = optuna.create_study(direction='maximize')
    study.optimize(_objective, n_trials=n_trials)

    best_params = study.best_params
    logger.info(f"[Optuna] Mejor AUC en prueba: {study.best_value:.4f} — params: {best_params}")

    # Entrenar modelo final con los mejores hiperparámetros sobre todos los datos SMOTE
    model = XGBClassifier(
        **best_params,
        random_state=42,
        eval_metric='logloss',
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_res, y_res)

    # 7. Evaluación del challenger
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    acc  = float(accuracy_score(y_test, y_pred))
    f1   = float(f1_score(y_test, y_pred, zero_division=0))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec  = float(recall_score(y_test, y_pred, zero_division=0))
    auc  = float(roc_auc_score(y_test, y_proba))
    training_time = time.time() - start_time
    logger.info(
        f"[Challenger] ACC={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} AUC={auc:.4f}"
    )

    challenger_metrics = {"auc_roc": auc, "f1_score": f1, "accuracy": acc,
                          "precision": prec, "recall": rec}

    # 8. Champion / Challenger
    champion_metrics: Optional[Dict] = None
    id_champion_model: Optional[int] = None
    try:
        with get_db_session() as session:
            champion = model_registry.get_current_champion(session)
            if champion:
                champion_metrics = model_registry.get_champion_metrics_from_audit(session, champion)
                id_champion_model = champion.id_model
    except Exception:
        logger.exception("[Churn] Error leyendo champion de BD")

    promoted, promotion_reason = _evaluate_promotion(challenger_metrics, champion_metrics)
    promotion_status = "PROMOTED" if promoted else "REJECTED"
    logger.info(f"[C/C] promovido={promoted} — {promotion_reason}")

    # 9. MLflow tracking
    mlflow_run_id = None
    version_tag   = f"churn_v_{int(time.time())}"
    try:
        init_dagshub_connection()
        mlflow.set_experiment("/churn_auto_training")
        with mlflow.start_run(run_name=version_tag) as run:
            for k, v in best_params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("train_samples",  len(X_train))
            mlflow.log_param("test_samples",   len(X_test))
            mlflow.log_param("smote",          True)
            mlflow.log_param("promoted",       promoted)
            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("f1_score",  f1)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall",    rec)
            mlflow.log_metric("auc_roc",   auc)
            mlflow.xgboost.log_model(model, "model")
            mlflow_run_id = run.info.run_id
            logger.info(f"[MLflow] Run registrado: {mlflow_run_id}")
    except Exception as e:
        logger.warning(f"[WARN] MLflow tracking fallo (no bloquea): {e}")

    # 10. Serializar combo-pack
    combo_pack = {
        'modelo_prediccion': model,
        'scaler':            scaler,
        'feature_names':     FEATURE_NAMES,
        'meta_info': {
            'version':       version_tag,
            'accuracy':      round(acc,  4),
            'f1_score':      round(f1,   4),
            'precision':     round(prec, 4),
            'recall':        round(rec,  4),
            'auc_roc':       round(auc,  4),
            'train_samples': len(X_train),
            'test_samples':  len(X_test),
            'mlflow_run_id': mlflow_run_id,
            'promoted':      promoted,
        }
    }
    buf = io.BytesIO()
    joblib.dump(combo_pack, buf)
    model_bytes = buf.getvalue()

    # 11. Si promovido: DagsHub upload + hot-reload
    dagshub_url:      Optional[str] = None
    dagshub_verified: bool          = False
    model_size_mb:    float         = 0.0

    if promoted:
        dagshub_url, model_size_mb = upload_champion(model_bytes, version_tag)
        if dagshub_url:
            dagshub_verified = verify_champion_integrity(version_tag)
        notify_hot_reload()   # best-effort

    # 12. Persistencia en BD
    model_id: Optional[int] = None
    try:
        model_config = {
            "algorithm":    "XGBoost + SMOTE",
            "best_params":  best_params,
            "feature_names": FEATURE_NAMES,
            "mlflow_run_id": mlflow_run_id,
        }
        with get_db_session() as session:
            dataset_id = model_registry.save_dataset_info(
                session=session,
                total_samples=total_samples,
                count_train=len(X_train),
                count_test=len(X_test),
                churn_ratio=churn_ratio,
                smote_applied=True,
            )
            model_id = model_registry.save_model_metadata(
                session=session,
                model_version=version_tag,
                algorithm="XGBoost + SMOTE",
                model_config=model_config,
                dagshub_url=dagshub_url,
                model_size_mb=model_size_mb if model_size_mb else None,
                promotion_status=promotion_status,
            )
            model_registry.save_complete_audit_record(
                session=session,
                id_dataset=dataset_id,
                id_model=model_id,
                start_training=datetime.fromtimestamp(start_time),
                end_training=datetime.now(),
                metrics=challenger_metrics,
                best_params=best_params,
                promotion_status=promotion_status,
                promotion_reason=promotion_reason,
                champion_metrics=champion_metrics,
                id_champion_model=id_champion_model,
                triggered_by=request.triggered_by,
                is_success=True,
            )
            session.commit()
    except Exception:
        logger.exception("[Churn] Error guardando en BD")
        return TrainingResponse(
            metrics=TrainingMetrics(
                accuracy=acc, f1_score=f1, precision=prec,
                recall=rec, auc_roc=auc, training_time_sec=round(training_time, 2),
            ),
            best_params=best_params,
            promotion_status="PERSISTENCE_ERROR",
            promotion_reason=promotion_reason,
            total_samples=total_samples,
            train_samples=len(X_train),
            test_samples=len(X_test),
            class_distribution={"0": int((y == 0).sum()), "1": int((y == 1).sum())},
            churn_ratio=round(churn_ratio, 4),
            model_version=version_tag,
            champion_metrics=champion_metrics,
            dagshub_verified=dagshub_verified,
            dagshub_url=dagshub_url,
            mlflow_run_id=mlflow_run_id,
        )

    # 13. Promover en BD si ganó
    if promoted and model_id is not None:
        try:
            with get_db_session() as session_promo:
                model_registry.promote_model_to_champion(
                    session=session_promo,
                    model_id=model_id,
                    promotion_reason=promotion_reason,
                )
                session_promo.commit()
        except Exception:
            logger.exception("[Churn] Error al promover en BD")

    elapsed = time.time() - start_time
    logger.info(f"=== AUTO-ENTRENAMIENTO CHURN FINALIZADO en {elapsed:.1f}s — {promotion_status} ===")

    return TrainingResponse(
        metrics=TrainingMetrics(
            accuracy=round(acc,  4),
            f1_score=round(f1,   4),
            precision=round(prec, 4),
            recall=round(rec,   4),
            auc_roc=round(auc,  4),
            training_time_sec=round(elapsed, 2),
        ),
        best_params=best_params,
        promotion_status=promotion_status,
        promotion_reason=promotion_reason,
        total_samples=total_samples,
        train_samples=len(X_train),
        test_samples=len(X_test),
        class_distribution={"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        churn_ratio=round(churn_ratio, 4),
        model_version=version_tag,
        champion_metrics=champion_metrics,
        dagshub_verified=dagshub_verified,
        dagshub_url=dagshub_url,
        mlflow_run_id=mlflow_run_id,
    )
