# src/fraude/training_service.py
import logging
import time
import base64
import io
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_curve,
    classification_report
)
from xgboost import XGBClassifier
import optuna
import shap

from fraude.fraude_schema import (
    TrainingRequest, TrainingResponse, TrainingMetrics, OptunaResult
)
from fraude.data_extraction import extract_and_balance_data, validate_date_range
from fraude.db_config import get_db_session
from fraude import model_registry
from fraude.dagshub_client import get_dagshub_client

logger = logging.getLogger(__name__)

# Features categóricas que necesitan encoding
CATEGORICAL_COLS = ['category', 'gender', 'job']

# Features numéricas que necesitan scaling
COLS_TO_SCALE = ['amt', 'city_pop', 'age', 'distance_km', 'hour']


def haversine_np(lon1, lat1, lon2, lat2):
    """Calcula distancia en km entre dos puntos geográficos usando fórmula Haversine."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def entrenar_modelo(request: TrainingRequest) -> TrainingResponse:
    """
    Pipeline completo de entrenamiento híbrido para fraude.
    
    NUEVO: Extrae datos automáticamente de la base de datos.
    
    Pipeline:
    1. Validar fechas
    2. Extraer + balance sampling desde BD
    3. Feature engineering
    4. Encoding y scaling
    5. IsolationForest (anomaly_score)
    6. Optuna optimization
    7. XGBoost final
    8. Threshold optimization
    9. Métricas
    10. SHAP explainer
    11. Serialización
    12. Response
    """
    print("=" * 80)
    print("🔍 DEBUG: INICIO DE entrenar_modelo() - CÓDIGO ACTUALIZADO CON PERSISTENCIA")
    print("=" * 80)
    
    start_time = time.time()
    logger.info(f"🚀 Iniciando autoentrenamiento de fraude")
    logger.info(f"   Fechas: {request.start_date} → {request.end_date}")
    logger.info(f"   Optuna trials: {request.optuna_trials}")
    logger.info(f"   Undersampling ratio: {request.undersampling_ratio}:1")


    # =========================================
    # 1. VALIDAR FECHAS
    # =========================================
    validate_date_range(request.start_date, request.end_date)

    
    # =========================================
    # 2. EXTRAER DATOS DE BD CON SAMPLING BALANCEADO
    # =========================================
    df = extract_and_balance_data(
        start_date=request.start_date,
        end_date=request.end_date,
        undersampling_ratio=request.undersampling_ratio
    )
    
    # Guardar ratio original para reporte
    fraud_count_original = len(df[df['is_fraud'] == 1])
    fraud_ratio_original = fraud_count_original / len(df)
    
    logger.info(f"📊 Datos extraídos: {len(df)} transacciones")
    logger.info(f"   Fraudes: {fraud_count_original} ({fraud_ratio_original:.1%})")
    
    # =========================================
    # 3. FEATURE ENGINEERING
    # =========================================
    logger.info("📐 Aplicando feature engineering...")
    
    # Calcular EDAD del cliente
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

    # Extraer HORA del día
    df['hour'] = df['trans_date_trans_time'].dt.hour

    # Calcular DISTANCIA entre cliente y comercio
    df['distance_km'] = haversine_np(
        df['long'], df['lat'], 
        df['merch_long'], df['merch_lat']
    )
    
    # Seleccionar features para entrenamiento
    feature_cols = ['amt', 'city_pop', 'category', 'gender', 'job', 
                    'age', 'hour', 'distance_km']
    
    X = df[feature_cols].copy()
    y = df['is_fraud'].copy()
    weights = df['sample_weight'].values
    
    # Distribución de clases
    class_dist = y.value_counts().to_dict()
    class_dist = {str(k): int(v) for k, v in class_dist.items()}
    fraud_ratio_balanced = class_dist.get('1', 0) / len(y)
    
    logger.info(f"📊 Distribución de clases: {class_dist}")
    logger.info(f"   Ratio de fraude balanceado: {fraud_ratio_balanced:.1%}")

    # =========================================
    # 4. ENCODING DE CATEGÓRICAS
    # =========================================
    logger.info("🔤 Encoding de variables categóricas...")
    
    encoders_dict = {}
    for col in CATEGORICAL_COLS:
        le_temp = LabelEncoder()
        le_temp.fit(X[col].astype(str))
        X[col] = le_temp.transform(X[col].astype(str))
        encoders_dict[col] = le_temp
    
    # =========================================
    # 5. SPLIT TRAIN/TEST
    # =========================================
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"📂 Train: {len(X_train)} | Test: {len(X_test)}")
    
    # =========================================
    # 6. SCALING
    # =========================================
    logger.info("⚖️ Aplicando scaling con RobustScaler...")
    
    scaler = RobustScaler()
    X_train[COLS_TO_SCALE] = scaler.fit_transform(X_train[COLS_TO_SCALE])
    X_test[COLS_TO_SCALE] = scaler.transform(X_test[COLS_TO_SCALE])
    
    logger.info("Datos procesados y listos para entrenamiento.")
    
    # =========================================
    # 7. ISOLATION FOREST (Detector de Anomalías)
    # =========================================
    logger.info("🌳 Entrenando Isolation Forest (contamination=0.005)...")
    
    if_model = IsolationForest(
        contamination=0.005,
        random_state=42,
        n_jobs=-1
    )
    if_model.fit(X_train)
    
    # Generar anomaly_score como nueva feature
    X_train['anomaly_score'] = if_model.decision_function(X_train)
    X_test['anomaly_score'] = if_model.decision_function(X_test)
    
    logger.info("✅ Anomaly scores generados")
    
    # =========================================
    # 8. OPTIMIZAR XGBOOST CON OPTUNA
    # =========================================
    logger.info(f"🔍 Iniciando optimización Optuna ({request.optuna_trials} trials)...")
    
    # Silenciar logs de Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        """Función objetivo: maximiza F1-Score"""
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 15, 35),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=request.optuna_trials)
    
    best_f1 = study.best_value
    best_params = study.best_params
    logger.info(f"✅ Mejor trial: F1-Score = {best_f1:.4f}")
    
    # =========================================
    # 9. ENTRENAR XGBOOST FINAL CON MEJORES PARÁMETROS
    # =========================================
    logger.info("🚀 Entrenando XGBoost final con los mejores parámetros...")
    
    xgb_model = XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    logger.info("✅ XGBoost entrenado con parámetros óptimos")
    
    # =========================================
    # 10. OPTIMIZACIÓN DE THRESHOLD
    # =========================================
    logger.info("🎯 Calculando threshold óptimo (Recall >= 95%)...")
    
    # Obtener probabilidades
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calcular curva Precision-Recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Estrategia: MAXIMIZAR Precision manteniendo Recall >= 95%
    target_recall = 0.95
    valid_indices = np.where(recalls >= target_recall)[0]
    
    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
        best_threshold = thresholds[best_idx]
        best_precision = precisions[best_idx]
        best_recall = recalls[best_idx]
    else:
        # Fallback
        best_threshold = 0.5
        best_precision = precision_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)
        best_recall = recall_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0)
    
    logger.info(f"🎯 Threshold óptimo: {best_threshold:.4f}")
    logger.info(f"   Recall esperado: {best_recall:.4f}")
    logger.info(f"   Precision esperada: {best_precision:.4f}")
    
    # Generar predicciones con threshold óptimo
    y_pred_optimizado = (y_prob >= best_threshold).astype(int)
    
    # =========================================
    # 11. CALCULAR MÉTRICAS
    # =========================================
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
        training_time_sec=round(training_time, 2)
    )
    
    logger.info(f"📊 Métricas: AUC={metrics.auc_roc} | F1={metrics.f1_score} | "
                f"Recall={metrics.recall} | Precision={metrics.precision}")
    
    # Imprimir reporte completo
    logger.info("\n--- Reporte de Clasificación ---")
    logger.info(f"\n{classification_report(y_test, y_pred_optimizado)}")
    
    # =========================================
    # 12. CREAR SHAP EXPLAINER
    # =========================================
    logger.info("🔬 Creando SHAP explainer...")
    
    try:
        explainer = shap.TreeExplainer(xgb_model)
        logger.info("✅ SHAP explainer creado correctamente")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo crear SHAP explainer: {e}")
        explainer = None
    
    # =========================================
    # 13. SERIALIZAR MODELO HÍBRIDO
    # =========================================
    logger.info("📦 Serializando modelo híbrido...")
    
    model_package = {
        'scaler': scaler,
        'model_xgb': xgb_model,
        'model_if': if_model,
        'encoders': encoders_dict,
        'explainer': explainer
    }
    
    buffer = io.BytesIO()
    joblib.dump(model_package, buffer)
    model_bytes = base64.b64encode(buffer.getvalue()).decode('utf-8')
    logger.info(f"📦 Modelo serializado: {len(model_bytes)} caracteres base64")
    
    # =========================================
    # 14. CONSTRUIR RESPONSE
    # =========================================
    
    # =========================================
    # 13.5. COMPARAR CON CHAMPION ACTUAL
    # =========================================
    logger.info("🏆 Comparando con modelo CHAMPION actual...")
    
    champion_metrics = None
    id_champion_model = None
    promotion_reason = None
    
    try:
        with get_db_session() as session:
            champion = model_registry.get_current_champion(session)
            
            if champion:
                # Hay un champion activo, comparar métricas
                logger.info(f"   Champion encontrado: {champion.model_version}")
                logger.info(f"   Champion threshold: {champion.threshold}")
                
                # Obtener métricas del champion desde el último audit
                champion_audit = session.query(model_registry.SelfTrainingAuditFraud).filter(
                    model_registry.SelfTrainingAuditFraud.id_model == champion.id_model
                ).order_by(model_registry.SelfTrainingAuditFraud.start_training.desc()).first()
                
                if champion_audit:
                    champion_metrics = {
                        'f1_score': float(champion_audit.f1_score) if champion_audit.f1_score else 0.0,
                        'recall': float(champion_audit.recall_score) if champion_audit.recall_score else 0.0,
                        'auc_roc': float(champion_audit.auc_roc) if champion_audit.auc_roc else 0.0
                    }
                    id_champion_model = champion.id_model
                    
                    logger.info(f"   Champion F1: {champion_metrics['f1_score']:.4f}")
                    logger.info(f"   Challenger F1: {metrics.f1_score:.4f}")
                    logger.info(f"   Champion Recall: {champion_metrics['recall']:.4f}")
                    logger.info(f"   Challenger Recall: {metrics.recall:.4f}")
                    
                    # CRITERIO DE PROMOCIÓN:
                    # 1. F1-Score debe ser >= que el champion
                    # 2. Recall debe ser >= que el champion (prioritario para fraude)
                    # 3. Si ambos son iguales o mejores → PROMOTED
                    
                    f1_diff = metrics.f1_score - champion_metrics['f1_score']
                    recall_diff = metrics.recall - champion_metrics['recall']
                    
                    if recall_diff >= 0 and f1_diff >= 0:
                        promotion_status = "PROMOTED"
                        promotion_reason = f"Mejor rendimiento: F1 +{f1_diff:.4f}, Recall +{recall_diff:.4f}"
                        logger.info(f"✅ PROMOTED: Challenger supera al champion")
                    elif recall_diff >= -0.01 and f1_diff > 0.005:
                        # Permite caída mínima de recall si F1 mejora significativamente
                        promotion_status = "PROMOTED"
                        promotion_reason = f"F1 mejorado significativamente (+{f1_diff:.4f}), Recall aceptable ({recall_diff:+.4f})"
                        logger.info(f"✅ PROMOTED: F1 mejora compensa ligera caída de recall")
                    else:
                        promotion_status = "REJECTED"
                        promotion_reason = f"Rendimiento insuficiente: F1 {f1_diff:+.4f}, Recall {recall_diff:+.4f}"
                        logger.info(f"❌ REJECTED: Challenger no supera al champion")
                else:
                    # Champion sin métricas → promover automáticamente
                    promotion_status = "PROMOTED"
                    promotion_reason = "Champion sin métricas registradas, promoción automática"
                    logger.warning("⚠️ Champion sin métricas, promoviendo challenger automáticamente")
                    
            else:
                # NO hay champion → primer modelo se promueve automáticamente
                promotion_status = "PROMOTED"
                promotion_reason = "Primer modelo entrenado, promoción automática a CHAMPION"
                logger.info("🎉 Primer modelo del sistema, promoción automática")
                
    except Exception as e:
        logger.error(f"❌ Error al comparar con champion: {e}")
        logger.error("   Marcando como PENDING para revisión manual")
        promotion_status = "PENDING"
        promotion_reason = f"Error en comparación: {str(e)}"
        import traceback
        traceback.print_exc()
    
    logger.info(f"📊 Decisión final: {promotion_status}")
    if promotion_reason:
        logger.info(f"   Razón: {promotion_reason}")

    model_config = {
        "architecture": "XGBoost + IsolationForest (Hybrid)",
        "strategy": "IF generates anomaly_score as feature for XGBoost",
        "xgboost_params": best_params,
        "isolation_forest_params": {
            "contamination": 0.005,
            "random_state": 42
        },
        "features_input": feature_cols,
        "features_derived": ["age", "hour", "distance_km", "anomaly_score"],
        "categorical_encoded": CATEGORICAL_COLS,
        "scaled_features": COLS_TO_SCALE,
        "optimal_threshold": float(best_threshold),
        "undersampling_ratio": request.undersampling_ratio,
        "date_range": f"{request.start_date} to {request.end_date}"
    }
    
    optuna_result = OptunaResult(
        best_trial_number=int(study.best_trial.number),
        best_f1_score=float(round(best_f1, 4)),
        best_params=best_params
    )
    
    print("🔍 DEBUG: Antes de crear TrainingResponse")
    
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
        fraud_ratio_balanced=float(round(fraud_ratio_balanced, 4))
    )
    
    print("🔍 DEBUG: Después de crear TrainingResponse")
    
    # =========================================
    # 15. PERSISTIR EN BASE DE DATOS
    # =========================================
    print("🔍 DEBUG: Llegó a la sección de persistencia")
    logger.info("💾 Guardando resultados en base de datos...")
    
    try:
        print("🔍 DEBUG: Entrando al try de guardado")
        with get_db_session() as session:
            print("🔍 DEBUG: Session creada")
            # 1. Guardar dataset info
            dataset_id = model_registry.save_dataset_info(
                session=session,
                start_date=request.start_date,
                end_date=request.end_date,
                total_samples=len(df),
                count_train=len(X_train),
                count_test=len(X_test),
                fraud_ratio=float(round(fraud_ratio_balanced, 4)),
                undersampling_ratio=request.undersampling_ratio
            )
            print(f"🔍 DEBUG: Dataset guardado, id={dataset_id}")
            
            # 2. Generar model_version único
            model_version = f"fraud_v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 3. Guardar model metadata (sin DagsHub URL por ahora)
            model_id = model_registry.save_model_metadata(
                session=session,
                model_version=model_version,
                algorithm="XGBoost + IsolationForest",
                model_config=model_config,
                threshold=float(best_threshold),
                promotion_status=promotion_status
            )
            print(f"🔍 DEBUG: Model guardado, id={model_id}")
            
            # 3.5 SUBIR MODELO A DAGSHUB
            print("🔍 DEBUG: Subiendo modelo a DagsHub...")
            logger.info("📤 Subiendo modelo a DagsHub...")
            
            dagshub_client = get_dagshub_client()
            dagshub_url, model_size_mb = dagshub_client.upload_model(
                model_bytes=model_bytes,
                model_version=model_version,
                artifact_path="fraud_models"
            )
            
            if dagshub_url:
                # Actualizar model metadata con URL y tamaño
                model_registry.update_model_dagshub_url(
                    session=session,
                    model_id=model_id,
                    dagshub_url=dagshub_url,
                    model_size_mb=model_size_mb
                )
                print(f"🔍 DEBUG: DagsHub URL actualizada: {dagshub_url}")
                logger.info(f"✅ Modelo subido a DagsHub: {dagshub_url}")
            else:
                logger.warning("⚠️ No se pudo subir modelo a DagsHub, continuando sin URL")
            
            # 4. Actualizar o crear audit record
            if request.audit_id:
                # Flujo Java: actualizar audit existente
                logger.info(f"📝 Actualizando audit record {request.audit_id} (flujo Java)")
                model_registry.update_audit_with_results(
                    session=session,
                    audit_id=request.audit_id,
                    id_dataset=dataset_id,
                    id_model=model_id,
                    end_training=datetime.now(),
                    metrics={
                        'accuracy': float(metrics.accuracy),
                        'precision': float(metrics.precision),
                        'recall': float(metrics.recall),
                        'f1_score': float(metrics.f1_score),
                        'auc_roc': float(metrics.auc_roc),
                        'optimal_threshold': float(metrics.optimal_threshold)
                    },
                    optuna_result={
                        'best_f1_score': float(optuna_result.best_f1_score),
                        'best_params': best_params
                    },
                    promotion_status=promotion_status,
                    promotion_reason=promotion_reason or f"Java training - {promotion_status}",
 id_champion_model=id_champion_model,
                    champion_metrics=champion_metrics,
                    is_success=True
                )
            else:
                # Flujo manual: crear audit completo de una vez con IDs válidos
                print("🔍 DEBUG: Creando audit completo (flujo manual)")
                logger.info("📝 Creando audit record completo (flujo manual)")
                
                audit_id = model_registry.save_complete_audit_record(
                    session=session,
                    id_dataset=dataset_id,
                    id_model=model_id,
                    start_training=datetime.fromtimestamp(start_time),
                    end_training=datetime.now(),
                    metrics={
                        'accuracy': float(metrics.accuracy),
                        'precision': float(metrics.precision),
                        'recall': float(metrics.recall),
                        'f1_score': float(metrics.f1_score),
                        'auc_roc': float(metrics.auc_roc),
                       'optimal_threshold': float(metrics.optimal_threshold)
                    },
                    optuna_result={
                        'best_f1_score': float(optuna_result.best_f1_score),
                        'best_params': best_params
                    },
                    promotion_status=promotion_status,
                    promotion_reason=promotion_reason or f"Manual training - {promotion_status}",
                    id_champion_model=id_champion_model,
                    champion_metrics=champion_metrics,
                    triggered_by=request.triggered_by,
                    is_success=True
                )
                print(f"🔍 DEBUG: Audit completo creado, id={audit_id}")
            
            session.commit()
            print("🔍 DEBUG: Session committed")
            logger.info(f"✅ Datos guardados: dataset_id={dataset_id}, model_id={model_id}, model_version={model_version}")
            
            # 5. Si fue PROMOTED, activar como CHAMPION y desactivar anterior
            if promotion_status == "PROMOTED":
                print("🔍 DEBUG: Promoviendo a CHAMPION...")
                logger.info("🏆 Promoviendo modelo a CHAMPION...")
                try:
                    success = model_registry.promote_to_champion(
                        session=session,
                        model_id=model_id,
                        promotion_reason=promotion_reason or "Promoted based on metrics"
                    )
                    if success:
                        session.commit()
                        print(f"🔍 DEBUG: Modelo {model_id} promovido a CHAMPION")
                        logger.info(f"✅ Modelo {model_id} activado como CHAMPION")
                    else:
                        logger.warning("⚠️ No se pudo promover el modelo")
                except Exception as promo_error:
                    logger.error(f"❌ Error al promover modelo: {promo_error}")
                    # No fallar el training por esto
                    import traceback
                    traceback.print_exc()
            
    except Exception as e:
        print(f"🔍 DEBUG: ERROR en guardado: {e}")
        logger.error(f"❌ Error guardando en BD: {e}")
        # No fallar el entrenamiento si falla el guardado
        import traceback
        traceback.print_exc()
    
    print("🔍 DEBUG: Salió del try-except de guardado")
    logger.info(f"✅ Entrenamiento completado en {training_time:.1f}s")
    return response

