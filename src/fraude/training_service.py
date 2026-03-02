# src/fraude/training_service.py
import io
import logging
import time
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import shap
import joblib
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

from fraude.data_extraction import extract_and_balance_data, validate_date_range
from fraude.db_config import get_db_session
from fraude.fraude_schema import (
    OptunaResult,
    TrainingMetrics,
    TrainingRequest,
    TrainingResponse,
)
from fraude import model_registry


logger = logging.getLogger(__name__)

# Features categóricas que necesitan encoding
CATEGORICAL_COLS = ["category", "gender", "job"]

# Features numéricas que necesitan scaling
COLS_TO_SCALE = ["amt", "city_pop", "age", "distance_km", "hour"]


# ---------------------------------------------------------------------------
# Utilidades de feature engineering
# ---------------------------------------------------------------------------

def haversine_np(lon1, lat1, lon2, lat2):
    """Calcula distancia en km entre dos puntos geográficos usando fórmula Haversine."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def entrenar_modelo(request: TrainingRequest) -> TrainingResponse:
    """
    Pipeline completo de entrenamiento híbrido para fraude.

    Extrae datos automáticamente de la base de datos.

    Pipeline:
    1.  Validar fechas
    2.  Extraer + balance sampling desde BD
    3.  Feature engineering
    4.  Encoding y scaling
    5.  IsolationForest (anomaly_score)
    6.  Optuna optimization
    7.  XGBoost final
    8.  Threshold optimization
    9.  Métricas
    10. SHAP explainer
    11. Serialización
    12. Comparación con champion
    13. Baseline PSI distributions
    14. Persistencia en BD (separada de la comparación)
    15. Promoción a CHAMPION (si aplica)
    """
    start_time = time.time()
    logger.info("🚀 Iniciando autoentrenamiento de fraude")
    logger.info("   Fechas: %s → %s", request.start_date, request.end_date)
    logger.info("   Optuna trials: %s", request.optuna_trials)
    logger.info("   Undersampling ratio: %s:1", request.undersampling_ratio)

    # =========================================================================
    # 1. VALIDAR FECHAS
    # =========================================================================
    validate_date_range(request.start_date, request.end_date)

    # =========================================================================
    # 2. EXTRAER DATOS DE BD CON SAMPLING BALANCEADO
    # =========================================================================
    df = extract_and_balance_data(
        start_date=request.start_date,
        end_date=request.end_date,
        undersampling_ratio=request.undersampling_ratio,
    )

    fraud_count_original = len(df[df["is_fraud"] == 1])
    fraud_ratio_original = fraud_count_original / len(df)

    logger.info("📊 Datos extraídos: %d transacciones", len(df))
    logger.info("   Fraudes: %d (%.1f%%)", fraud_count_original, fraud_ratio_original * 100)

    # =========================================================================
    # 3. FEATURE ENGINEERING
    # =========================================================================
    logger.info("📐 Aplicando feature engineering...")

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # relativedelta calcula años exactos considerando bisiestos
    df["age"] = df.apply(
        lambda r: relativedelta(r["trans_date_trans_time"], r["dob"]).years, axis=1
    )
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["distance_km"] = haversine_np(df["long"], df["lat"], df["merch_long"], df["merch_lat"])

    feature_cols = ["amt", "city_pop", "category", "gender", "job", "age", "hour", "distance_km"]

    X = df[feature_cols].copy()
    y = df["is_fraud"].copy()
    weights = df["sample_weight"].values

    class_dist = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}
    fraud_ratio_balanced = class_dist.get("1", 0) / len(y)

    logger.info("📊 Distribución de clases: %s", class_dist)
    logger.info("   Ratio de fraude balanceado: %.1f%%", fraud_ratio_balanced * 100)

    # =========================================================================
    # 4. ENCODING DE CATEGÓRICAS
    # =========================================================================
    logger.info("🔤 Encoding de variables categóricas...")

    encoders_dict = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        X[col] = le.transform(X[col].astype(str))
        encoders_dict[col] = le

    # =========================================================================
    # 5. SPLIT TRAIN/TEST
    # =========================================================================
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("📂 Train: %d | Test: %d", len(X_train), len(X_test))

    # =========================================================================
    # 6. SCALING
    # =========================================================================
    logger.info("⚖️ Aplicando scaling con RobustScaler...")

    scaler = RobustScaler()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])

    # =========================================================================
    # 7. ISOLATION FOREST (Detector de Anomalías)
    # =========================================================================
    logger.info("🌳 Entrenando Isolation Forest (contamination=0.005)...")

    if_model = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
    if_model.fit(X_train)

    X_train["anomaly_score"] = if_model.decision_function(X_train)
    X_test["anomaly_score"] = if_model.decision_function(X_test)

    logger.info("✅ Anomaly scores generados")

    # =========================================================================
    # 8. OPTIMIZAR XGBOOST CON OPTUNA
    # =========================================================================
    logger.info("🔍 Iniciando optimización Optuna (%d trials)...", request.optuna_trials)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        """Función objetivo: maximiza F1-Score"""
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 15, 35),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "random_state": 42,
            "eval_metric": "logloss",
            # use_label_encoder eliminado: parámetro obsoleto desde XGBoost 1.6
        }
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=request.optuna_trials)

    best_f1 = study.best_value
    best_params = study.best_params
    logger.info("✅ Mejor trial: F1-Score = %.4f", best_f1)

    # =========================================================================
    # 9. ENTRENAR XGBOOST FINAL CON MEJORES PARÁMETROS
    # =========================================================================
    logger.info("🚀 Entrenando XGBoost final con los mejores parámetros...")

    xgb_model = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        # use_label_encoder eliminado: parámetro obsoleto desde XGBoost 1.6
    )
    xgb_model.fit(X_train, y_train)
    logger.info("✅ XGBoost entrenado con parámetros óptimos")

    # =========================================================================
    # 10. OPTIMIZACIÓN DE THRESHOLD
    # =========================================================================
    logger.info("🎯 Calculando threshold óptimo (Recall >= 95%)...")

    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    target_recall = 0.95
    valid_indices = np.where(recalls >= target_recall)[0]

    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
        best_threshold = thresholds[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
    else:
        best_threshold = 0.5
        best_precision = precision_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)
        best_recall = recall_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)

    logger.info("🎯 Threshold óptimo: %.4f", best_threshold)
    logger.info("   Recall esperado: %.4f", best_recall)
    logger.info("   Precision esperada: %.4f", best_precision)

    y_pred_optimizado = (y_prob >= best_threshold).astype(int)

    # =========================================================================
    # 11. CALCULAR MÉTRICAS
    # =========================================================================
    logger.info("📏 Calculando métricas finales...")

    auc_roc = roc_auc_score(y_test, y_prob, sample_weight=w_test)
    training_time = time.time() - start_time

    metrics = TrainingMetrics(
        auc_roc=round(auc_roc, 4),
        accuracy=round(accuracy_score(y_test, y_pred_optimizado, sample_weight=w_test), 4),
        precision=round(precision_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0), 4),
        recall=round(recall_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0), 4),
        f1_score=round(f1_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0), 4),
        optimal_threshold=round(best_threshold, 4),
        training_time_sec=round(training_time, 2),
    )

    logger.info(
        "📊 Métricas: AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f",
        metrics.auc_roc, metrics.f1_score, metrics.recall, metrics.precision,
    )
    logger.info("\n--- Reporte de Clasificación ---\n%s", classification_report(y_test, y_pred_optimizado))

    # =========================================================================
    # 12. CREAR SHAP EXPLAINER
    # =========================================================================
    logger.info("🔬 Creando SHAP explainer...")

    try:
        explainer = shap.TreeExplainer(xgb_model)
        logger.info("✅ SHAP explainer creado correctamente")
    except Exception:
        logger.warning("⚠️ No se pudo crear SHAP explainer", exc_info=True)
        explainer = None

    # =========================================================================
    # 13. SERIALIZAR MODELO HÍBRIDO
    # =========================================================================
    logger.info("📦 Serializando modelo híbrido...")

    model_package = {
        "scaler": scaler,
        "model_xgb": xgb_model,
        "model_if": if_model,
        "encoders": encoders_dict,
        "explainer": explainer,
    }

    buffer = io.BytesIO()
    joblib.dump(model_package, buffer)
    # Extraer bytes inmediatamente para no depender del estado del buffer más tarde
    raw_model_bytes = buffer.getvalue()
    model_bytes = __import__("base64").b64encode(raw_model_bytes).decode("utf-8")
    logger.info("📦 Modelo serializado: %d caracteres base64", len(model_bytes))

    # =========================================================================
    # 14. COMPARAR CON CHAMPION ACTUAL
    # =========================================================================
    logger.info("🏆 Comparando con modelo CHAMPION actual...")

    champion_metrics = None
    id_champion_model = None
    promotion_reason = None
    promotion_status = "PENDING"

    try:
        with get_db_session() as session:
            champion = model_registry.get_current_champion(session)

            if champion:
                logger.info("   Champion encontrado: %s", champion.model_version)

                champion_audit = session.query(model_registry.SelfTrainingAuditFraud).filter(
                    model_registry.SelfTrainingAuditFraud.id_model == champion.id_model
                ).order_by(model_registry.SelfTrainingAuditFraud.start_training.desc()).first()

                if champion_audit:
                    champion_metrics = {
                        "f1_score": float(champion_audit.f1_score) if champion_audit.f1_score else 0.0,
                        "recall": float(champion_audit.recall_score) if champion_audit.recall_score else 0.0,
                        "auc_roc": float(champion_audit.auc_roc) if champion_audit.auc_roc else 0.0,
                    }
                    id_champion_model = champion.id_model

                    f1_diff = metrics.f1_score - champion_metrics["f1_score"]
                    recall_diff = metrics.recall - champion_metrics["recall"]

                    logger.info("   Champion F1=%.4f | Challenger F1=%.4f", champion_metrics["f1_score"], metrics.f1_score)
                    logger.info("   Champion Recall=%.4f | Challenger Recall=%.4f", champion_metrics["recall"], metrics.recall)

                    if recall_diff >= 0 and f1_diff >= 0:
                        promotion_status = "PROMOTED"
                        promotion_reason = f"Mejor rendimiento: F1 +{f1_diff:.4f}, Recall +{recall_diff:.4f}"
                        logger.info("✅ PROMOTED: Challenger supera al champion")
                    elif recall_diff >= -0.01 and f1_diff > 0.005:
                        promotion_status = "PROMOTED"
                        promotion_reason = f"F1 mejorado (+{f1_diff:.4f}), Recall aceptable ({recall_diff:+.4f})"
                        logger.info("✅ PROMOTED: F1 mejora compensa ligera caída de recall")
                    else:
                        promotion_status = "REJECTED"
                        promotion_reason = f"Rendimiento insuficiente: F1 {f1_diff:+.4f}, Recall {recall_diff:+.4f}"
                        logger.info("❌ REJECTED: Challenger no supera al champion")
                else:
                    promotion_status = "PROMOTED"
                    promotion_reason = "Champion sin métricas registradas, promoción automática"
                    logger.warning("⚠️ Champion sin métricas, promoviendo challenger automáticamente")
            else:
                promotion_status = "PROMOTED"
                promotion_reason = "Primer modelo entrenado, promoción automática a CHAMPION"
                logger.info("🎉 Primer modelo del sistema, promoción automática")

    except Exception:
        logger.exception("❌ Error al comparar con champion. Marcando como PENDING.")
        promotion_status = "PENDING"
        promotion_reason = "Error en comparación con champion"

    logger.info("📊 Decisión final: %s — %s", promotion_status, promotion_reason)

    # =========================================================================
    # 15. BASELINE DISTRIBUTIONS (para PSI Drift)
    # =========================================================================
    logger.info("📐 Calculando baseline_distributions para PSI drift...")

    baseline_distributions = {}
    NUMERIC_FEATURES_FOR_PSI = ["amt", "city_pop", "age", "distance_km", "hour"]
    PERCENTILES = list(range(10, 100, 10))

    for feat in NUMERIC_FEATURES_FOR_PSI:
        try:
            if feat not in df.columns:
                logger.warning("   Feature '%s' no encontrada en df, omitiendo baseline.", feat)
                continue
            values = df[feat].dropna().values
            if len(values) == 0:
                continue
            pct_edges = np.percentile(values, PERCENTILES).tolist()
            bins = [float(values.min())] + pct_edges + [float(values.max())]
            counts, _ = np.histogram(values, bins=np.unique(bins))
            total = counts.sum()
            pct = (counts / total * 100.0).tolist() if total > 0 else [0.0] * len(counts)
            baseline_distributions[feat] = {
                "bins": [round(b, 6) for b in np.unique(bins).tolist()],
                "pct": [round(p, 6) for p in pct],
                "n_samples": int(total),
            }
            logger.info("   ✅ Baseline '%s': %d bins, N=%d", feat, len(pct), total)
        except Exception:
            logger.warning("   ⚠️ No se pudo calcular baseline para '%s'", feat, exc_info=True)

    logger.info("📐 Baseline calculado para %d features", len(baseline_distributions))

    model_config = {
        "architecture": "XGBoost + IsolationForest (Hybrid)",
        "strategy": "IF generates anomaly_score as feature for XGBoost",
        "xgboost_params": best_params,
        "isolation_forest_params": {"contamination": 0.005, "random_state": 42},
        "features_input": feature_cols,
        "features_derived": ["age", "hour", "distance_km", "anomaly_score"],
        "categorical_encoded": CATEGORICAL_COLS,
        "scaled_features": COLS_TO_SCALE,
        "optimal_threshold": float(best_threshold),
        "undersampling_ratio": request.undersampling_ratio,
        "date_range": f"{request.start_date} to {request.end_date}",
        "baseline_distributions": baseline_distributions,
    }

    optuna_result = OptunaResult(
        best_trial_number=int(study.best_trial.number),
        best_f1_score=float(round(best_f1, 4)),
        best_params=best_params,
    )

    response = TrainingResponse(
        metrics=metrics,
        optuna_result=optuna_result,
        model_base64=model_bytes,
        model_config_dict=model_config,
        promotion_status=promotion_status,
        total_samples=len(df),
        train_samples=len(X_train),
        test_samples=len(X_test),
        class_distribution=class_dist,
        fraud_ratio_balanced=float(round(fraud_ratio_balanced, 4)),
    )

    # =========================================================================
    # 16. PERSISTIR EN BASE DE DATOS
    # Nota: la subida a DagsHub ocurre FUERA del session de BD para no
    #       mantener conexiones abiertas durante el upload (puede tardar 30-60s).
    # =========================================================================
    logger.info("💾 Guardando resultados en base de datos...")

    model_version = f"fraud_v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_id = None

    try:
        with get_db_session() as session:
            # 1. Dataset info
            dataset_id = model_registry.save_dataset_info(
                session=session,
                start_date=request.start_date,
                end_date=request.end_date,
                total_samples=len(df),
                count_train=len(X_train),
                count_test=len(X_test),
                fraud_ratio=float(round(fraud_ratio_balanced, 4)),
                undersampling_ratio=request.undersampling_ratio,
            )

            # 2. Model metadata
            model_id = model_registry.save_model_metadata(
                session=session,
                model_version=model_version,
                algorithm="XGBoost + IsolationForest",
                model_config=model_config,
                threshold=float(best_threshold),
                promotion_status=promotion_status,
            )

            # 3. Audit record
            if request.audit_id:
                logger.info("📝 Actualizando audit record %s (flujo Java)", request.audit_id)
                model_registry.update_audit_with_results(
                    session=session,
                    audit_id=request.audit_id,
                    id_dataset=dataset_id,
                    id_model=model_id,
                    end_training=datetime.now(),
                    metrics={
                        "accuracy": float(metrics.accuracy),
                        "precision": float(metrics.precision),
                        "recall": float(metrics.recall),
                        "f1_score": float(metrics.f1_score),
                        "auc_roc": float(metrics.auc_roc),
                        "optimal_threshold": float(metrics.optimal_threshold),
                    },
                    optuna_result={
                        "trials": request.optuna_trials,
                        "best_trial_number": int(optuna_result.best_trial_number),
                        "best_f1_score": float(optuna_result.best_f1_score),
                        "best_params": best_params,
                    },
                    promotion_status=promotion_status,
                    promotion_reason=promotion_reason or f"Java training - {promotion_status}",
                    id_champion_model=id_champion_model,
                    champion_metrics=champion_metrics,
                    is_success=True,
                )
            else:
                logger.info("📝 Creando audit record completo (flujo manual)")
                model_registry.save_complete_audit_record(
                    session=session,
                    id_dataset=dataset_id,
                    id_model=model_id,
                    start_training=datetime.fromtimestamp(start_time),
                    end_training=datetime.now(),
                    metrics={
                        "accuracy": float(metrics.accuracy),
                        "precision": float(metrics.precision),
                        "recall": float(metrics.recall),
                        "f1_score": float(metrics.f1_score),
                        "auc_roc": float(metrics.auc_roc),
                        "optimal_threshold": float(metrics.optimal_threshold),
                    },
                    optuna_result={
                        "trials": request.optuna_trials,
                        "best_trial_number": int(optuna_result.best_trial_number),
                        "best_f1_score": float(optuna_result.best_f1_score),
                        "best_params": best_params,
                    },
                    promotion_status=promotion_status,
                    promotion_reason=promotion_reason or f"Manual training - {promotion_status}",
                    id_champion_model=id_champion_model,
                    champion_metrics=champion_metrics,
                    triggered_by=request.triggered_by,
                    is_success=True,
                )

            # Un único commit para dataset + model + audit
            session.commit()
            logger.info(
                "✅ Datos guardados: dataset_id=%s, model_id=%s, model_version=%s",
                dataset_id, model_id, model_version,
            )

    except Exception:
        logger.exception("❌ Error guardando en BD. El entrenamiento fue exitoso pero no se persistió.")
        response.promotion_status = "PERSISTENCE_ERROR"
        return response

    # =========================================================================
    # 17. SUBIR MODELO A DAGSHUB (fuera del session de BD)
    # =========================================================================
    if model_id is not None:
        logger.info("📤 Subiendo modelo a DagsHub...")
        try:
            from fraude.dagshub_client import upload_champion as dagshub_upload

            dagshub_url, model_size_mb = dagshub_upload(
                model_bytes=raw_model_bytes,
                version_tag=model_version,
            )

            if dagshub_url:
                logger.info("✅ Modelo subido a DagsHub: %s", dagshub_url)
                # Actualizar URL con una sesión nueva, independiente del commit anterior
                try:
                    with get_db_session() as session_url:
                        model_registry.update_model_dagshub_url(
                            session=session_url,
                            model_id=model_id,
                            dagshub_url=dagshub_url,
                            model_size_mb=model_size_mb,
                        )
                        session_url.commit()
                except Exception:
                    logger.warning("⚠️ No se pudo actualizar URL DagsHub en BD", exc_info=True)
            else:
                logger.warning("⚠️ DagsHub no devolvió URL. Modelo no vinculado.")

        except Exception:
            logger.warning("⚠️ No se pudo subir modelo a DagsHub. Continuando.", exc_info=True)

    # =========================================================================
    # 18. PROMOCIÓN A CHAMPION (sesión separada — después del commit de datos)
    # =========================================================================
    if promotion_status == "PROMOTED" and model_id is not None:
        logger.info("🏆 Promoviendo modelo a CHAMPION...")
        try:
            with get_db_session() as session_promo:
                success = model_registry.promote_model_to_champion(
                    session=session_promo,
                    model_id=model_id,
                    promotion_reason=promotion_reason or "Promoted based on metrics",
                )
                if success:
                    session_promo.commit()
                    logger.info("✅ Modelo %s activado como CHAMPION", model_id)
                else:
                    logger.warning("⚠️ promote_model_to_champion retornó False")
        except Exception:
            logger.exception("❌ Error al promover modelo a CHAMPION")
            # No sobreescribir promotion_status — el modelo ya está guardado con PROMOTED en BD

    training_time_total = time.time() - start_time
    logger.info("✅ Entrenamiento completado en %.1fs", training_time_total)
    return response
