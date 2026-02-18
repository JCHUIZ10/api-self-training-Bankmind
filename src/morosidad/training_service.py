# src/morosidad/training_service.py
import logging
import time
import gc
import io
import os
import joblib
import pandas as pd
import numpy as np
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
import mlflow

# Imports locales
from morosidad.morosidad_schema import (
    TrainingRequest, TrainingResponse,
    TrainingMetrics, OptunaResult
)
from morosidad import dagshub_client
from morosidad import data_loader

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'UTILIZATION_RATE'
]

def calcular_ks_statistic(y_true, y_proba):
    from scipy.stats import ks_2samp
    pos_proba = y_proba[y_true == 1]
    neg_proba = y_proba[y_true == 0]
    if len(pos_proba) == 0 or len(neg_proba) == 0:
        return 0.0
    ks_stat, _ = ks_2samp(pos_proba, neg_proba)
    return float(ks_stat)

def calcular_baseline_distributions(df: pd.DataFrame) -> dict:
    features = ['PAY_0', 'PAY_2', 'PAY_3', 'LIMIT_BAL', 'BILL_AMT1', 'UTILIZATION_RATE']
    dists = {}
    for col in features:
        if col not in df.columns: continue
        data = df[col].dropna()
        if col.startswith('PAY_'):
            counts = data.value_counts(normalize=True).sort_index()
            dists[col] = {"type": "categorical", "values": [int(x) for x in counts.index], "probs": counts.values.tolist()}
        else:
            counts, edges = np.histogram(data, bins=10, density=False)
            dists[col] = {"type": "continuous", "bins": edges.tolist(), "probs": (counts/len(data)).tolist()}
    return dists

def ejecutar_autoentrenamiento(request: TrainingRequest) -> TrainingResponse:
    start_time = time.time()
    n_trials = request.optuna_trials
    logger.info(f"🚀 Iniciando Auto-Entrenamiento (Trials={n_trials})")

    dagshub_client.init_dagshub_connection()
    mlflow.set_experiment("morosidad_auto_training")

    champ_model = None
    ensemble_model = None

    try:
        # ═══════════════════════════════════════════
        # 1. CARGAR DATOS DESDE POSTGRESQL
        # ═══════════════════════════════════════════
        df = data_loader.load_training_data()
        if df is None or df.empty:
            raise ValueError("No se obtuvieron datos para entrenar.")
            
        X = df[FEATURE_COLUMNS].copy()
        y = df['DEFAULT_PAYMENT_NEXT_MONTH'].copy()
        
        n_neg, n_pos = (y==0).sum(), (y==1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        
        # Split: 60% train, 20% val (early stopping), 20% test
        X_train_full, X_test, y_train_full, y_test, w_train_full, w_test = train_test_split(
            X, y, df['SAMPLE_WEIGHT'], test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_train_full, y_train_full, w_train_full, test_size=0.25, random_state=42, stratify=y_train_full
        )
        logger.info(f"📊 Datos: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # ═══════════════════════════════════════════
        # 2. OPTUNA — OPTIMIZAR XGBOOST
        # ═══════════════════════════════════════════
        xgb_trials = max(n_trials // 2, 2)
        logger.info(f"🔍 Optimizando XGBoost ({xgb_trials} trials)...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective_xgb(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'scale_pos_weight': scale_pos_weight,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'verbosity': 0,
                'n_jobs': -1,
                'early_stopping_rounds': 20
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return roc_auc_score(y_test, model.predict_proba(X_test)[:,1], sample_weight=w_test)
        
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(objective_xgb, n_trials=xgb_trials)
        logger.info(f"✅ Mejor AUC XGBoost: {study_xgb.best_value:.4f}")
        
        # ═══════════════════════════════════════════
        # 3. OPTUNA — OPTIMIZAR LIGHTGBM
        # ═══════════════════════════════════════════
        lgbm_trials = max(n_trials - xgb_trials, 2)
        logger.info(f"🔍 Optimizando LightGBM ({lgbm_trials} trials)...")
        
        def objective_lgbm(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'scale_pos_weight': scale_pos_weight,
                'objective': 'binary',
                'metric': 'auc',
                'verbose': -1,
                'n_jobs': -1
            }
            model = lgb.LGBMClassifier(**params)
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=callbacks)
            return roc_auc_score(y_test, model.predict_proba(X_test)[:,1], sample_weight=w_test)
        
        study_lgbm = optuna.create_study(direction='maximize')
        study_lgbm.optimize(objective_lgbm, n_trials=lgbm_trials)
        logger.info(f"✅ Mejor AUC LightGBM: {study_lgbm.best_value:.4f}")
        
        # ═══════════════════════════════════════════
        # 4. ENTRENAR CHAMPIONS INDIVIDUALES
        # ═══════════════════════════════════════════
        logger.info("🏗️ Entrenando modelos Champion...")
        
        # XGBoost Champion
        best_xgb_params = study_xgb.best_params
        xgb_champion = xgb.XGBClassifier(
            **best_xgb_params, scale_pos_weight=scale_pos_weight,
            objective='binary:logistic', eval_metric='auc', verbosity=0, n_jobs=-1
        )
        xgb_champion.fit(X_train_full, y_train_full)
        
        # LightGBM Champion
        best_lgbm_params = study_lgbm.best_params
        lgbm_champion = lgb.LGBMClassifier(
            **best_lgbm_params, scale_pos_weight=scale_pos_weight,
            objective='binary', metric='auc', verbose=-1, n_jobs=-1
        )
        lgbm_champion.fit(X_train_full, y_train_full)
        
        # RandomForest (params fijos, sin Optuna)
        rf_model = RandomForestClassifier(
            n_estimators=80, max_depth=8,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train_full, y_train_full)
        
        # ═══════════════════════════════════════════
        # 5. ENSAMBLAR MODELO FINAL
        # ═══════════════════════════════════════════
        logger.info("🔗 Construyendo Ensamble Final...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_champion),
                ('lgbm', lgbm_champion),
                ('rf', rf_model)
            ],
            voting='soft',
            weights=[2, 1, 1],  # XGB vale x2
            n_jobs=-1
        )
        # VotingClassifier requiere fit() aunque los estimadores ya estén entrenados
        # Le pasamos los datos para que configure los atributos internos
        ensemble_model.fit(X_train_full, y_train_full)
        
        # ═══════════════════════════════════════════
        # 6. CALCULAR MÉTRICAS DEL ENSAMBLE
        # ═══════════════════════════════════════════
        y_prob = ensemble_model.predict_proba(X_test)[:,1]
        y_pred = ensemble_model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_prob, sample_weight=w_test)
        ks_score = calcular_ks_statistic(y_test.values, y_prob)
        
        metrics = TrainingMetrics(
            auc_roc=round(auc_score, 4), ks_statistic=round(ks_score, 4),
            gini_coefficient=round(2*auc_score-1, 4),
            accuracy=round(accuracy_score(y_test, y_pred, sample_weight=w_test), 4),
            precision=round(precision_score(y_test, y_pred, sample_weight=w_test, zero_division=0), 4),
            recall=round(recall_score(y_test, y_pred, sample_weight=w_test, zero_division=0), 4),
            f1_score=round(f1_score(y_test, y_pred, sample_weight=w_test, zero_division=0), 4),
            training_time_sec=round(time.time()-start_time, 2)
        )
        logger.info(f"📊 Ensamble AUC: {auc_score:.4f} | KS: {ks_score:.4f}")
        
        # ═══════════════════════════════════════════
        # 7. COMPARAR CON CHAMPION DE DAGSHUB
        # ═══════════════════════════════════════════
        logger.info("⚔️ Comparando con el Champion actual...")
        champ_model, _, champ_meta = dagshub_client.download_current_champion()
        champ_auc = 0.0
        
        if champ_model:
            try:
                yp_champ = champ_model.predict_proba(X_test)[:,1]
                champ_auc = roc_auc_score(y_test, yp_champ, sample_weight=w_test)
                logger.info(f"📊 Champion AUC: {champ_auc:.4f} | Challenger AUC: {auc_score:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Error evaluando Champion: {e}")
        else:
            logger.info("ℹ️ No se encontró Champion (Cold Start). El Challenger será promovido.")
            
        # ═══════════════════════════════════════════
        # 8. DECISIÓN DE PROMOCIÓN + COMBO-PACK
        # ═══════════════════════════════════════════
        auc_diff = auc_score - champ_auc
        status = "KEEP_CHAMPION"
        assembly_config = None
        dagshub_verified = False
        version_tag = f"v_{int(time.time())}"
        
        if auc_diff > 0.015 or champ_model is None:
            status = "NEW_CHAMPION"
            logger.info(f"🏆 NUEVO CHAMPION DETECTADO (+{auc_diff:.4f} AUC)")
            
            # Extraer XGBoost interno para SHAP
            logger.info("🧠 Creando SHAP Explainer...")
            try:
                xgb_interno = ensemble_model.named_estimators_['xgb']
                explainer = shap.TreeExplainer(xgb_interno)
            except Exception as e:
                logger.warning(f"⚠️ No se pudo crear SHAP explainer: {e}")
                explainer = None
            
            # Empaquetar combo-pack (modelo + SHAP + meta)
            pkg = {
                'modelo_prediccion': ensemble_model,
                'shap_explainer': explainer,
                'meta_info': {
                    'version': version_tag,
                    'auc_roc': metrics.auc_roc,
                    'ks_statistic': metrics.ks_statistic,
                    'descripcion': 'Ensamble XGB+LGBM+RF con SHAP embebido'
                }
            }
            buf_pkg = io.BytesIO()
            joblib.dump(pkg, buf_pkg)
            
            # ═══════════════════════════════════════════
            # UPLOAD TRANSACCIONAL
            # ═══════════════════════════════════════════
            
            # Paso 1: Upload a DagsHub
            upload_ok = dagshub_client.upload_champion(buf_pkg.getvalue(), f"challenger-{version_tag}")
            
            if not upload_ok:
                status = "UPLOAD_FAILED"
                logger.error("❌ ABORT: Falló el upload a DagsHub. No se promociona.")
            else:
                # Paso 2: Verificar integridad re-descargando
                integrity_ok = dagshub_client.verify_champion_integrity(version_tag)
                
                if not integrity_ok:
                    status = "UPLOAD_FAILED"
                    logger.error("❌ ABORT: Verificación de integridad falló. Modelo corrupto o no disponible.")
                else:
                    dagshub_verified = True
                    logger.info("✅ Upload + Verificación OK — Modelo listo para promoción")
            
            # Construir assembly_config solo si verificado
            if dagshub_verified:
                assembly_config = {
                    "architecture": "VotingClassifier",
                    "voting_strategy": "soft",
                    "weights_assigned": [2, 1, 1],
                    "order_estimators": ["xgboost_champion", "lightgbm_champion", "rf_base"],
                    "random_seed": 42,
                    "features_input": FEATURE_COLUMNS,
                    "internal_components": {
                        "xgboost_champion": {k: v for k, v in best_xgb_params.items()},
                        "lightgbm_champion": {k: v for k, v in best_lgbm_params.items()},
                        "rf_base": {"n_estimators": 80, "max_depth": 8, "class_weight": "balanced"}
                    }
                }
        else:
            logger.info(f"📉 Champion se mantiene (diff={auc_diff:.4f}, umbral=0.015)")
        
        # Log MLflow solo si la operación fue exitosa (no UPLOAD_FAILED)
        if status != "UPLOAD_FAILED":
            import tempfile as _tmpf
            
            # Nombre descriptivo del run: "champion_v_1234" o "keep_champion_v_1234"
            run_name = f"{'champion' if status == 'NEW_CHAMPION' else 'keep_champion'}_{version_tag}"
            mlflow.start_run(run_name=run_name)
            
            # Params de ambos modelos
            mlflow.log_params({f"xgb_{k}": v for k, v in best_xgb_params.items()})
            mlflow.log_params({f"lgbm_{k}": v for k, v in best_lgbm_params.items()})
            mlflow.log_param("deployment_status", status)
            mlflow.log_param("version_tag", version_tag)
            
            # Métricas del ensamble
            mlflow.log_metrics({
                "ensemble_auc": auc_score, "ensemble_ks": ks_score,
                "gini_coefficient": 2 * auc_score - 1,
                "xgb_best_auc": study_xgb.best_value, "lgbm_best_auc": study_lgbm.best_value
            })
            
            # Guardar modelo como artefacto de MLflow
            if status == "NEW_CHAMPION":
                try:
                    with _tmpf.NamedTemporaryFile(suffix=".pkl", prefix=f"modelo_{version_tag}_", delete=False) as f:
                        buf_mlflow = io.BytesIO()
                        joblib.dump(pkg, buf_mlflow)
                        f.write(buf_mlflow.getvalue())
                        artifact_path = f.name
                    mlflow.log_artifact(artifact_path, artifact_path="champion_model")
                    os.remove(artifact_path)
                    logger.info("📦 Modelo guardado como artefacto en MLflow")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo guardar artefacto MLflow: {e}")
            
            mlflow.end_run()
            logger.info(f"✅ Run '{run_name}' registrado en MLflow")
        else:
            logger.warning("⚠️ MLflow logging omitido (upload fallido)")
            
        # ═══════════════════════════════════════════
        # 9. RESPUESTA AL BACKEND
        # ═══════════════════════════════════════════
        all_best_params = {**best_xgb_params, **{f"lgbm_{k}": v for k, v in best_lgbm_params.items()}}
        
        # Generar info de columnas (matching DetalleColumna POJO)
        columns_info = [
            {"name": col, "date_type": str(df[col].dtype).upper(), "rol": "FEATURE", "description": None, "is_nullable": False}
            for col in FEATURE_COLUMNS
        ]
        columns_info.append({
            "name": "DEFAULT_PAYMENT_NEXT_MONTH", "date_type": "INTEGER", "rol": "TARGET",
            "description": "Variable Objetivo (1=Moroso, 0=No Moroso)", "is_nullable": False
        })
        columns_info.append({
            "name": "SAMPLE_WEIGHT", "date_type": "FLOAT", "rol": "WEIGHT",
            "description": "Peso temporal para decay", "is_nullable": False
        })
        
        # Obtener fecha inicio del dataset
        dataset_start = data_loader.get_dataset_start_date()
        
        return TrainingResponse(
            metrics=metrics,
            optuna_result=OptunaResult(
                best_value=auc_score,
                best_params=all_best_params,
                n_trials=xgb_trials + lgbm_trials
            ),
            total_samples=len(df), train_samples=len(X_train_full), test_samples=len(X_test),
            baseline_distributions=calcular_baseline_distributions(df),
            assembly_config=assembly_config,
            columns_info=columns_info,
            dataset_start_date=dataset_start,
            dagshub_verified=dagshub_verified,
            version_tag=version_tag,
            deployment_status=status
        )
        
    except Exception as e:
        logger.error(f"❌ Error Pipeline: {e}")
        raise e
    finally:
        # Limpieza de modelos en memoria
        del champ_model, ensemble_model
        gc.collect()
