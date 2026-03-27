import io
import time
import math
import base64
import logging
from datetime import datetime

import joblib
from sklearn.model_selection import train_test_split

from fraude.schemas.fraude import TrainingRequest, TrainingResponse, OptunaResult
from fraude.core.utils import get_reference_date, compute_lambda, validate_training_dates
from fraude.data.data_extraction import extract_training_data, DataProvider
from fraude.data.db_config import get_db_session
from fraude.data import model_registry

from fraude.core.training.feature_engineering import apply_feature_engineering
from fraude.core.training.preprocessing import encode_categorical_features, scale_numeric_features, CATEGORICAL_COLS, COLS_TO_SCALE
from fraude.core.training.model_trainer import train_isolation_forest, optimize_and_train_xgboost
from fraude.core.training.model_evaluator import evaluate_model, create_shap_explainer, compute_baseline_distributions
from fraude.core.training.model_promoter import evaluate_promotion

logger = logging.getLogger(__name__)

def entrenar_modelo(request: TrainingRequest) -> TrainingResponse:
    start_time = time.time()
    logger.info("🚀 Iniciando autoentrenamiento de fraude con decay temporal (Refactorizado)")

    reference_dt = get_reference_date()
    end_date = request.end_date or reference_dt.strftime("%Y-%m-%d")
    start_date = request.start_date
    lam = compute_lambda(request.half_life_days)
    
    validate_training_dates(end_date=end_date, start_date=start_date)

    df_raw = extract_training_data(
        end_date=end_date, lam=lam, max_history_days=request.max_history_days,
        undersampling_ratio=request.undersampling_ratio, start_date=start_date
    )
    
    df = apply_feature_engineering(df_raw)
    
    provider = DataProvider(df, if_recent_months=request.if_recent_months)
    df_full = provider.get_full_data()
    df_recent = provider.get_recent_data()
    
    feature_cols = ["amt", "city_pop", "category", "gender", "job", "age", "hour", "distance_km"]
    X = df_full[feature_cols].copy()
    y = df_full["is_fraud"].copy()
    weights = df_full["sample_weight"].values
    
    class_dist = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}
    fraud_ratio_balanced = class_dist.get("1", 0) / len(y)
    
    X_recent = df_recent[feature_cols].copy()
    
    X, X_recent, encoders_dict = encode_categorical_features(X, X_recent)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test, X_recent_scaled, scaler = scale_numeric_features(X_train, X_test, X_recent)
    
    if_model = train_isolation_forest(X_recent_scaled)
    X_train["anomaly_score"] = if_model.decision_function(X_train)
    X_test["anomaly_score"] = if_model.decision_function(X_test)
    
    xgb_model, best_params, best_f1, best_trial_num = optimize_and_train_xgboost(
        X_train, y_train, w_train, X_test, y_test, request.optuna_trials
    )
    
    metrics, best_threshold = evaluate_model(xgb_model, X_test, y_test, w_test, start_time)
    explainer = create_shap_explainer(xgb_model)
    baseline_distributions = compute_baseline_distributions(df, ["amt", "city_pop", "age", "distance_km", "hour"])
    
    # Serializar modelo
    logger.info("📦 Serializando modelo híbrido...")
    model_package = {
        "scaler": scaler, "model_xgb": xgb_model, "model_if": if_model,
        "encoders": encoders_dict, "explainer": explainer,
    }
    buffer = io.BytesIO()
    joblib.dump(model_package, buffer)
    raw_model_bytes = buffer.getvalue()
    model_bytes = base64.b64encode(raw_model_bytes).decode("utf-8")
    
    # Comparar Champion
    champion_metrics = None
    id_champion_model = None
    try:
        with get_db_session() as session:
            champion = model_registry.get_current_champion(session)
            if champion:
                champion_audit = (
                    session.query(model_registry.SelfTrainingAuditFraud)
                    .filter(model_registry.SelfTrainingAuditFraud.id_model == champion.id_model)
                    .order_by(model_registry.SelfTrainingAuditFraud.start_training.desc())
                    .first()
                )
                if champion_audit:
                    champion_metrics = {
                        "f1_score": float(champion_audit.f1_score) if champion_audit.f1_score else 0.0,
                        "recall": float(champion_audit.recall_score) if champion_audit.recall_score else 0.0,
                        "auc_roc": float(champion_audit.auc_roc) if champion_audit.auc_roc else 0.0,
                    }
                    id_champion_model = champion.id_model
    except Exception:
        logger.exception("❌ Error obteniendo champion DB")

    metrics_dict = {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "auc_roc": metrics.auc_roc,
        "optimal_threshold": float(best_threshold)
    }
    promotion_status, promotion_reason = evaluate_promotion(metrics_dict, champion_metrics)

    ts_min = df["trans_date_trans_time"].min()
    ts_max = df["trans_date_trans_time"].max()
    effective_date_range = f"{ts_min.date()} → {ts_max.date()}"
    model_version = f"fraud_v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model_config = {
        "architecture": "XGBoost + IsolationForest (Hybrid Refactored)",
        "strategy": "IF genera anomaly_score como feature para XGBoost",
        "xgboost_params": best_params,
        "temporal_decay": {"half_life_days": request.half_life_days, "lambda": round(lam, 6)},
        "features_input": feature_cols,
        "optimal_threshold": float(best_threshold),
        "undersampling_ratio": request.undersampling_ratio,
        "baseline_distributions": baseline_distributions
    }
    
    optuna_result = OptunaResult(
        best_trial_number=int(best_trial_num), best_f1_score=float(best_f1), best_params=best_params
    )
    
    response = TrainingResponse(
        metrics=metrics, optuna_result=optuna_result, model_base64=model_bytes,
        model_config_dict=model_config, promotion_status=promotion_status,
        total_samples=len(df_full), train_samples=len(X_train), test_samples=len(X_test),
        class_distribution=class_dist, fraud_ratio_balanced=float(round(fraud_ratio_balanced, 4)),
        half_life_days=request.half_life_days, effective_date_range=effective_date_range
    )
    
    # DB Persistence
    model_id = None
    try:
        with get_db_session() as session:
            dataset_id = model_registry.save_dataset_info(
                session=session, start_date=str(ts_min.date()), end_date=str(ts_max.date()),
                total_samples=len(df_full), count_train=len(X_train), count_test=len(X_test),
                fraud_ratio=float(round(fraud_ratio_balanced, 4)), undersampling_ratio=request.undersampling_ratio
            )
            model_id = model_registry.save_model_metadata(
                session=session, model_version=model_version, algorithm="XGBoost + IsolationForest",
                model_config=model_config, threshold=float(best_threshold), promotion_status=promotion_status
            )
            if request.audit_id:
                model_registry.update_audit_with_results(
                    session=session, audit_id=request.audit_id, id_dataset=dataset_id, id_model=model_id,
                    end_training=datetime.now(), metrics=metrics_dict, optuna_result={
                        "trials": request.optuna_trials, "best_trial_number": int(optuna_result.best_trial_number),
                        "best_f1_score": float(optuna_result.best_f1_score), "best_params": best_params
                    }, promotion_status=promotion_status, promotion_reason=promotion_reason,
                    id_champion_model=id_champion_model, champion_metrics=champion_metrics, is_success=True
                )
            else:
                model_registry.save_complete_audit_record(
                    session=session, id_dataset=dataset_id, id_model=model_id,
                    start_training=datetime.fromtimestamp(start_time), end_training=datetime.now(),
                    metrics=metrics_dict, optuna_result={
                        "trials": request.optuna_trials, "best_trial_number": int(optuna_result.best_trial_number),
                        "best_f1_score": float(optuna_result.best_f1_score), "best_params": best_params
                    }, promotion_status=promotion_status, promotion_reason=promotion_reason,
                    id_champion_model=id_champion_model, champion_metrics=champion_metrics,
                    triggered_by=request.triggered_by, is_success=True
                )
            session.commit()
    except Exception:
        logger.exception("[ERROR] Error guardando en BD.")
        response.promotion_status = "PERSISTENCE_ERROR"
        return response

    # DagsHub
    if model_id is not None:
        try:
            from fraude.infrastructure.dagshub import upload_champion as dagshub_upload
            dagshub_url, model_size_mb = dagshub_upload(model_bytes=raw_model_bytes, version_tag=model_version)
            if dagshub_url:
                with get_db_session() as session_url:
                    model_registry.update_model_dagshub_url(session=session_url, model_id=model_id, dagshub_url=dagshub_url, model_size_mb=model_size_mb)
                    session_url.commit()
        except Exception:
            logger.warning("⚠️ No se pudo subir modelo a DagsHub", exc_info=True)

    # Promover Champion
    if promotion_status == "PROMOTED" and model_id is not None:
        try:
            with get_db_session() as session_promo:
                success = model_registry.promote_model_to_champion(session=session_promo, model_id=model_id, promotion_reason=promotion_reason)
                if success:
                    session_promo.commit()
        except Exception:
            logger.exception("❌ Error al promover modelo a CHAMPION")

    logger.info("✅ Autoentrenamiento completado en %.1fs", time.time() - start_time)
    return response
