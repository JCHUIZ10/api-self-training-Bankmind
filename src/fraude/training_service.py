# src/fraude/training_service.py
import io
import logging
import math
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

from fraude.data_extraction import (
    DataProvider,
    compute_lambda,
    extract_training_data,
    get_reference_date,
    validate_training_dates,
)
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


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def haversine_np(lon1, lat1, lon2, lat2):
    """Calcula distancia en km entre dos puntos geográficos (fórmula Haversine)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica feature engineering sobre un DataFrame extraído de la BD.
    Devuelve el mismo DataFrame con columnas adicionales: age, hour, distance_km.
    La columna trans_date_trans_time se convierte a datetime in-place.
    """
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"]                   = pd.to_datetime(df["dob"])
    df["age"]         = df.apply(
        lambda r: relativedelta(r["trans_date_trans_time"], r["dob"]).years, axis=1
    )
    df["hour"]        = df["trans_date_trans_time"].dt.hour
    df["distance_km"] = haversine_np(df["long"], df["lat"], df["merch_long"], df["merch_lat"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# XGBObjective — Callable para Optuna con sample_weight
# ─────────────────────────────────────────────────────────────────────────────

class XGBObjective:
    """
    Función objetivo inyectable para Optuna.

    Encapsula los datos de entrenamiento/test y los pesos temporales, de modo
    que Optuna optimiza hiperparámetros ponderando correctamente los datos
    históricos (datos viejos valen menos que datos recientes).

    Maximiza F1-Score sobre el conjunto de test.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        w_train: np.ndarray,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.w_train = w_train

    def __call__(self, trial: optuna.Trial) -> float:
        param = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
            "max_depth":        trial.suggest_int("max_depth", 4, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.05, 0.2),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 15, 35),
            "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
            "random_state":     42,
            "eval_metric":      "logloss",
        }
        model = XGBClassifier(**param)
        # Los pesos temporales se pasan a fit para que Optuna elija
        # hiperparámetros que priorizan los datos más recientes.
        model.fit(self.X_train, self.y_train, sample_weight=self.w_train)
        preds = model.predict(self.X_test)
        return f1_score(self.y_test, preds)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_modelo(request: TrainingRequest) -> TrainingResponse:
    """
    Pipeline completo de entrenamiento híbrido para fraude con decay temporal.

    Flujo:
     1.  Resolver fechas de referencia (env var REFERENCE_DATE o now())
     2.  Calcular λ desde half_life_days
     3.  Validar fechas
     4.  Extraer datos con decay exponencial calculado en SQL
     5.  Feature engineering (age, hour, distance_km)
     6.  DataProvider: vistas por modelo (full para XGB, reciente para IF)
     7.  Encoding de categóricas
     8.  Split train/test (estratificado)
     9.  Scaling (RobustScaler)
    10.  IsolationForest — entrenado solo con datos recientes (últimos N meses)
    11.  Optuna con XGBObjective (pasa sample_weight → hiperparámetros correctos)
    12.  XGBoost final con sample_weight
    13.  Threshold optimization (Recall >= 95%)
    14.  Métricas finales
    15.  SHAP explainer
    16.  Serialización del modelo híbrido
    17.  Comparación champion vs challenger
    18.  Baseline PSI distributions
    19.  Persistencia en BD
    20.  Upload a DagsHub
    21.  Promoción a CHAMPION (si aplica)
    """
    start_time = time.time()
    logger.info("🚀 Iniciando autoentrenamiento de fraude con decay temporal")

    # =========================================================================
    # 1. RESOLVER FECHAS DE REFERENCIA
    # =========================================================================
    reference_dt = get_reference_date()
    end_date     = request.end_date or reference_dt.strftime("%Y-%m-%d")
    start_date   = request.start_date  # puede ser None → se calcula en extract

    logger.info("   Fecha de referencia (end_date): %s", end_date)
    logger.info("   start_date explícito           : %s", start_date or "None (automático)")
    logger.info("   max_history_days               : %d días", request.max_history_days)
    logger.info("   half_life_days                 : %d días", request.half_life_days)
    logger.info("   if_recent_months               : %d meses (IsolationForest)", request.if_recent_months)
    logger.info("   Optuna trials                  : %d", request.optuna_trials)
    logger.info("   Undersampling ratio            : %d:1", request.undersampling_ratio)

    # =========================================================================
    # 2. CALCULAR λ DESDE HALF_LIFE_DAYS
    # =========================================================================
    lam = compute_lambda(request.half_life_days)

    # =========================================================================
    # 3. VALIDAR FECHAS
    # =========================================================================
    validate_training_dates(end_date=end_date, start_date=start_date)

    # =========================================================================
    # 4. EXTRAER DATOS CON DECAY EXPONENCIAL EN SQL
    # =========================================================================
    df_raw = extract_training_data(
        end_date          = end_date,
        lam               = lam,
        max_history_days  = request.max_history_days,
        undersampling_ratio = request.undersampling_ratio,
        start_date        = start_date,
    )

    fraud_count_original = int((df_raw["is_fraud"] == 1).sum())
    fraud_ratio_original = fraud_count_original / len(df_raw)
    logger.info("📊 Datos extraídos: %d transacciones", len(df_raw))
    logger.info("   Fraudes: %d (%.1f%%)", fraud_count_original, fraud_ratio_original * 100)

    # =========================================================================
    # 5. FEATURE ENGINEERING (sobre todo el dataset)
    # =========================================================================
    logger.info("📐 Aplicando feature engineering...")
    df = apply_feature_engineering(df_raw)

    # =========================================================================
    # 6. DATAPROVIDER — separación de responsabilidades
    # =========================================================================
    provider = DataProvider(df, if_recent_months=request.if_recent_months)

    # Dataset completo para XGBoost (todas las épocas con pesos)
    df_full   = provider.get_full_data()
    # Dataset reciente para IsolationForest (sin pesos, contexto actual)
    df_recent = provider.get_recent_data()
    logger.info("   DataProvider full  : %d registros", len(df_full))
    logger.info("   DataProvider recent: %d registros (IF)", len(df_recent))

    # =========================================================================
    # 7. PREPARAR FEATURES / LABELS / PESOS
    # =========================================================================
    feature_cols = ["amt", "city_pop", "category", "gender", "job", "age", "hour", "distance_km"]

    X       = df_full[feature_cols].copy()
    y       = df_full["is_fraud"].copy()
    weights = df_full["sample_weight"].values

    class_dist         = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}
    fraud_ratio_balanced = class_dist.get("1", 0) / len(y)
    logger.info("📊 Distribución de clases: %s", class_dist)
    logger.info("   Ratio de fraude balanceado: %.1f%%", fraud_ratio_balanced * 100)

    # =========================================================================
    # 8. ENCODING DE CATEGÓRICAS
    # =========================================================================
    logger.info("🔤 Encoding de variables categóricas...")
    encoders_dict = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        X[col] = le.transform(X[col].astype(str))
        encoders_dict[col] = le

    # Aplicar el mismo encoding al subset reciente para IF
    X_recent = df_recent[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        le = encoders_dict[col]
        # Valores no vistos se reemplazan por la primera clase conocida (fallback seguro)
        X_recent[col] = X_recent[col].astype(str).apply(
            lambda v: v if v in le.classes_ else le.classes_[0]
        )
        X_recent[col] = le.transform(X_recent[col])

    # =========================================================================
    # 9. SPLIT TRAIN / TEST (estratificado)
    # =========================================================================
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("📂 Train: %d | Test: %d", len(X_train), len(X_test))
    logger.info(
        "   sample_weight train — min=%.4f, mean=%.4f, max=%.4f",
        w_train.min(), w_train.mean(), w_train.max(),
    )

    # =========================================================================
    # 10. SCALING
    # =========================================================================
    logger.info("⚖️ Aplicando scaling con RobustScaler...")
    scaler = RobustScaler()
    X_train[COLS_TO_SCALE]  = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE]   = scaler.transform(X_test[COLS_TO_SCALE])

    # Escalar también el subset reciente para IF
    X_recent_scaled = X_recent.copy()
    X_recent_scaled[COLS_TO_SCALE] = scaler.transform(X_recent[COLS_TO_SCALE])

    # =========================================================================
    # 11. ISOLATION FOREST — solo datos recientes
    # =========================================================================
    logger.info(
        "🌳 Entrenando Isolation Forest con %d registros (últimos %d meses)...",
        len(X_recent_scaled), request.if_recent_months,
    )
    if_model = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
    if_model.fit(X_recent_scaled)

    # Generar anomaly_score para train y test usando el IF entrenado en datos recientes
    X_train["anomaly_score"] = if_model.decision_function(X_train)
    X_test["anomaly_score"]  = if_model.decision_function(X_test)
    logger.info("✅ Anomaly scores generados (IF entrenado en contexto actual)")

    # =========================================================================
    # 12. OPTIMIZACIÓN OPTUNA CON SAMPLE_WEIGHT
    # =========================================================================
    logger.info(
        "🔍 Iniciando optimización Optuna (%d trials) con sample_weight...",
        request.optuna_trials,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    objective = XGBObjective(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        w_train=w_train,
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=request.optuna_trials)

    best_f1     = study.best_value
    best_params = study.best_params
    logger.info("✅ Mejor trial: F1-Score = %.4f", best_f1)

    # =========================================================================
    # 13. ENTRENAR XGBOOST FINAL CON MEJORES PARÁMETROS + SAMPLE_WEIGHT
    # =========================================================================
    logger.info("🚀 Entrenando XGBoost final con parámetros óptimos y sample_weight (%d muestras)...", len(X_train))
    xgb_model = XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    logger.info("✅ XGBoost entrenado con pesos temporales")

    # =========================================================================
    # 14. OPTIMIZACIÓN DE THRESHOLD (Recall >= 95%)
    # =========================================================================
    logger.info("🎯 Calculando threshold óptimo (Recall >= 95%)...")
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    target_recall  = 0.95
    valid_indices  = np.where(recalls >= target_recall)[0]

    if len(valid_indices) > 0:
        best_idx       = valid_indices[np.argmax(precisions[valid_indices])]
        best_threshold = thresholds[best_idx]
        best_precision = precisions[best_idx]
        best_recall    = recalls[best_idx]
    else:
        best_threshold = 0.5
        best_precision = precision_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)
        best_recall    = recall_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)

    logger.info("🎯 Threshold óptimo: %.4f", best_threshold)
    logger.info("   Recall esperado  : %.4f", best_recall)
    logger.info("   Precision esperada: %.4f", best_precision)

    y_pred_optimizado = (y_prob >= best_threshold).astype(int)

    # =========================================================================
    # 15. MÉTRICAS FINALES
    # =========================================================================
    logger.info("📏 Calculando métricas finales...")
    auc_roc       = roc_auc_score(y_test, y_prob, sample_weight=w_test)
    training_time = time.time() - start_time

    metrics = TrainingMetrics(
        auc_roc           = round(auc_roc, 4),
        accuracy          = round(accuracy_score(y_test, y_pred_optimizado, sample_weight=w_test), 4),
        precision         = round(precision_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0), 4),
        recall            = round(recall_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0), 4),
        f1_score          = round(f1_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0), 4),
        optimal_threshold = round(best_threshold, 4),
        training_time_sec = round(training_time, 2),
    )
    logger.info(
        "📊 Métricas: AUC=%.4f | F1=%.4f | Recall=%.4f | Precision=%.4f",
        metrics.auc_roc, metrics.f1_score, metrics.recall, metrics.precision,
    )
    logger.info("\n--- Reporte de Clasificación ---\n%s", classification_report(y_test, y_pred_optimizado))

    # =========================================================================
    # 16. SHAP EXPLAINER
    # =========================================================================
    logger.info("🔬 Creando SHAP explainer...")
    try:
        explainer = shap.TreeExplainer(xgb_model)
        logger.info("✅ SHAP explainer creado correctamente")
    except Exception:
        logger.warning("⚠️ No se pudo crear SHAP explainer", exc_info=True)
        explainer = None

    # =========================================================================
    # 17. SERIALIZAR MODELO HÍBRIDO
    # =========================================================================
    logger.info("📦 Serializando modelo híbrido...")
    model_package = {
        "scaler":    scaler,
        "model_xgb": xgb_model,
        "model_if":  if_model,
        "encoders":  encoders_dict,
        "explainer": explainer,
    }
    buffer         = io.BytesIO()
    joblib.dump(model_package, buffer)
    raw_model_bytes = buffer.getvalue()
    model_bytes     = __import__("base64").b64encode(raw_model_bytes).decode("utf-8")
    logger.info("📦 Modelo serializado: %d caracteres base64", len(model_bytes))

    # =========================================================================
    # 18. COMPARAR CON CHAMPION ACTUAL
    # =========================================================================
    logger.info("🏆 Comparando con modelo CHAMPION actual...")
    champion_metrics  = None
    id_champion_model = None
    promotion_reason  = None
    promotion_status  = "PENDING"

    try:
        with get_db_session() as session:
            champion = model_registry.get_current_champion(session)

            if champion:
                logger.info("   Champion encontrado: %s", champion.model_version)
                champion_audit = (
                    session.query(model_registry.SelfTrainingAuditFraud)
                    .filter(model_registry.SelfTrainingAuditFraud.id_model == champion.id_model)
                    .order_by(model_registry.SelfTrainingAuditFraud.start_training.desc())
                    .first()
                )
                if champion_audit:
                    champion_metrics = {
                        "f1_score": float(champion_audit.f1_score)      if champion_audit.f1_score      else 0.0,
                        "recall":   float(champion_audit.recall_score)  if champion_audit.recall_score  else 0.0,
                        "auc_roc":  float(champion_audit.auc_roc)       if champion_audit.auc_roc       else 0.0,
                    }
                    id_champion_model = champion.id_model

                    f1_diff     = metrics.f1_score - champion_metrics["f1_score"]
                    recall_diff = metrics.recall   - champion_metrics["recall"]

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
    # 19. BASELINE DISTRIBUTIONS (para PSI Drift)
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
            bins      = [float(values.min())] + pct_edges + [float(values.max())]
            counts, _ = np.histogram(values, bins=np.unique(bins))
            total     = counts.sum()
            pct       = (counts / total * 100.0).tolist() if total > 0 else [0.0] * len(counts)
            baseline_distributions[feat] = {
                "bins":      [round(b, 6) for b in np.unique(bins).tolist()],
                "pct":       [round(p, 6) for p in pct],
                "n_samples": int(total),
            }
            logger.info("   ✅ Baseline '%s': %d bins, N=%d", feat, len(pct), total)
        except Exception:
            logger.warning("   ⚠️ No se pudo calcular baseline para '%s'", feat, exc_info=True)

    logger.info("📐 Baseline calculado para %d features", len(baseline_distributions))

    # Rango efectivo de fechas (extraído del propio DataFrame para trazabilidad)
    ts_min = df["trans_date_trans_time"].min()
    ts_max = df["trans_date_trans_time"].max()
    effective_date_range = f"{ts_min.date()} → {ts_max.date()}"

    model_version = f"fraud_v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_config  = {
        "architecture":          "XGBoost + IsolationForest (Hybrid)",
        "strategy":              "IF genera anomaly_score como feature para XGBoost",
        "xgboost_params":        best_params,
        "isolation_forest_params": {
            "contamination":  0.005,
            "random_state":   42,
            "training_window": f"últimos {request.if_recent_months} meses (datos recientes)"
        },
        "temporal_decay": {
            "half_life_days":  request.half_life_days,
            "lambda":          round(lam, 6),
            "weight_at_1year": round(math.exp(-lam * 365), 4),
        },
        "features_input":        feature_cols,
        "features_derived":      ["age", "hour", "distance_km", "anomaly_score"],
        "categorical_encoded":   CATEGORICAL_COLS,
        "scaled_features":       COLS_TO_SCALE,
        "optimal_threshold":     float(best_threshold),
        "undersampling_ratio":   request.undersampling_ratio,
        "effective_date_range":  effective_date_range,
        "max_history_days":      request.max_history_days,
        "baseline_distributions": baseline_distributions,
    }

    optuna_result = OptunaResult(
        best_trial_number = int(study.best_trial.number),
        best_f1_score     = float(round(best_f1, 4)),
        best_params       = best_params,
    )

    response = TrainingResponse(
        metrics              = metrics,
        optuna_result        = optuna_result,
        model_base64         = model_bytes,
        model_config_dict    = model_config,
        promotion_status     = promotion_status,
        total_samples        = len(df_full),
        train_samples        = len(X_train),
        test_samples         = len(X_test),
        class_distribution   = class_dist,
        fraud_ratio_balanced = float(round(fraud_ratio_balanced, 4)),
        half_life_days       = request.half_life_days,
        effective_date_range = effective_date_range,
    )

    # =========================================================================
    # 20. PERSISTIR EN BASE DE DATOS
    # =========================================================================
    logger.info("💾 Guardando resultados en base de datos...")
    model_id = None

    try:
        with get_db_session() as session:
            dataset_id = model_registry.save_dataset_info(
                session          = session,
                start_date       = str(ts_min.date()),
                end_date         = str(ts_max.date()),
                total_samples    = len(df_full),
                count_train      = len(X_train),
                count_test       = len(X_test),
                fraud_ratio      = float(round(fraud_ratio_balanced, 4)),
                undersampling_ratio = request.undersampling_ratio,
            )

            model_id = model_registry.save_model_metadata(
                session          = session,
                model_version    = model_version,
                algorithm        = "XGBoost + IsolationForest",
                model_config     = model_config,
                threshold        = float(best_threshold),
                promotion_status = promotion_status,
            )

            if request.audit_id:
                logger.info("📝 Actualizando audit record %s (flujo Java)", request.audit_id)
                model_registry.update_audit_with_results(
                    session          = session,
                    audit_id         = request.audit_id,
                    id_dataset       = dataset_id,
                    id_model         = model_id,
                    end_training     = datetime.now(),
                    metrics          = {
                        "accuracy":          float(metrics.accuracy),
                        "precision":         float(metrics.precision),
                        "recall":            float(metrics.recall),
                        "f1_score":          float(metrics.f1_score),
                        "auc_roc":           float(metrics.auc_roc),
                        "optimal_threshold": float(metrics.optimal_threshold),
                    },
                    optuna_result    = {
                        "trials":             request.optuna_trials,
                        "best_trial_number":  int(optuna_result.best_trial_number),
                        "best_f1_score":      float(optuna_result.best_f1_score),
                        "best_params":        best_params,
                    },
                    promotion_status = promotion_status,
                    promotion_reason = promotion_reason or f"Java training - {promotion_status}",
                    id_champion_model = id_champion_model,
                    champion_metrics = champion_metrics,
                    is_success       = True,
                )
            else:
                logger.info("📝 Creando audit record completo (flujo manual)")
                model_registry.save_complete_audit_record(
                    session          = session,
                    id_dataset       = dataset_id,
                    id_model         = model_id,
                    start_training   = datetime.fromtimestamp(start_time),
                    end_training     = datetime.now(),
                    metrics          = {
                        "accuracy":          float(metrics.accuracy),
                        "precision":         float(metrics.precision),
                        "recall":            float(metrics.recall),
                        "f1_score":          float(metrics.f1_score),
                        "auc_roc":           float(metrics.auc_roc),
                        "optimal_threshold": float(metrics.optimal_threshold),
                    },
                    optuna_result    = {
                        "trials":             request.optuna_trials,
                        "best_trial_number":  int(optuna_result.best_trial_number),
                        "best_f1_score":      float(optuna_result.best_f1_score),
                        "best_params":        best_params,
                    },
                    promotion_status = promotion_status,
                    promotion_reason = promotion_reason or f"Manual training - {promotion_status}",
                    id_champion_model = id_champion_model,
                    champion_metrics = champion_metrics,
                    triggered_by     = request.triggered_by,
                    is_success       = True,
                )

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
    # 21. SUBIR MODELO A DAGSHUB (fuera del session de BD)
    # =========================================================================
    if model_id is not None:
        logger.info("📤 Subiendo modelo a DagsHub...")
        try:
            from fraude.dagshub_client import upload_champion as dagshub_upload
            dagshub_url, model_size_mb = dagshub_upload(
                model_bytes = raw_model_bytes,
                version_tag = model_version,
            )
            if dagshub_url:
                logger.info("✅ Modelo subido a DagsHub: %s", dagshub_url)
                try:
                    with get_db_session() as session_url:
                        model_registry.update_model_dagshub_url(
                            session      = session_url,
                            model_id     = model_id,
                            dagshub_url  = dagshub_url,
                            model_size_mb = model_size_mb,
                        )
                        session_url.commit()
                except Exception:
                    logger.warning("⚠️ No se pudo actualizar URL DagsHub en BD", exc_info=True)
            else:
                logger.warning("⚠️ DagsHub no devolvió URL. Modelo no vinculado.")
        except Exception:
            logger.warning("⚠️ No se pudo subir modelo a DagsHub. Continuando.", exc_info=True)

    # =========================================================================
    # 22. PROMOCIÓN A CHAMPION (sesión independiente)
    # =========================================================================
    if promotion_status == "PROMOTED" and model_id is not None:
        logger.info("🏆 Promoviendo modelo a CHAMPION...")
        try:
            with get_db_session() as session_promo:
                success = model_registry.promote_model_to_champion(
                    session          = session_promo,
                    model_id         = model_id,
                    promotion_reason = promotion_reason or "Promoted based on metrics",
                )
                if success:
                    session_promo.commit()
                    logger.info("✅ Modelo %s activado como CHAMPION", model_id)
                else:
                    logger.warning("⚠️ promote_model_to_champion retornó False")
        except Exception:
            logger.exception("❌ Error al promover modelo a CHAMPION")

    total_training_time = time.time() - start_time
    logger.info("✅ Autoentrenamiento completado en %.1fs", total_training_time)
    logger.info("   Rango efectivo: %s", effective_date_range)
    logger.info("   Decay: half_life=%d días, λ=%.6f", request.half_life_days, lam)
    return response
