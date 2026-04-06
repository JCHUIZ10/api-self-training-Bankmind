# src/retiro_atm/training_service.py
import logging
import time
import os
from datetime import datetime, date

import mlflow
import requests

from retiro_atm import database
from retiro_atm.schemas import (
    TrainingRequest,
    TrainingResponse,
    TrainingMetrics,
    ConfidenceInterval,
    DatasetWithdrawalPrediction,
    SelfTrainingAuditWithdrawalModel,
    WithdrawalModel,
)
from retiro_atm.self_train.data_loader import load_dataset, obtener_distribucion_actual_atm_features, consultar_ultima_version_modelo
from retiro_atm.self_train.data_preprocessor import DataPreprocessor, FEATURES, TARGET
from retiro_atm.self_train.calculate_psi import get_psi
from retiro_atm.self_train.model_optimizer import ModelOptimizer
from retiro_atm.self_train.model_evaluator import ModelEvaluator
from retiro_atm.self_train.dagshub_client import AtmModelProvider
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# URL del backend Java para notificar actualización de modelo
UPDATE_MODEL_API_URL = os.getenv("UPDATE_MODEL_API_URL")

def ejecutar_autoentrenamiento(request: TrainingRequest) -> TrainingResponse:
    """
    Orquesta el pipeline completo de autoentrenamiento:
    1. Cargar datos → 2. Registrar dataset → 3. Preprocesar → 4. Calcular PSI →
    5. Optimizar → 6. Entrenar → 7. Evaluar → 8. Comparar con champion →
    9. Promover si es mejor → 10. Retornar respuesta.
    """
    start_time = time.time()
    start_datetime = datetime.now()
    session = None

    try:
        # ─── 1. Cargar datos ───
        logger.info("═══ PASO 1/10: Cargando datos ═══")
        df = load_dataset()

        # ─── 2. Preprocesar ───
        logger.info("═══ PASO 2/10: Preprocesando datos ═══")
        data = DataPreprocessor.preparar_datos_completos(
            df,
            dias_test=request.dias_particion_test,
            dias_val=request.dias_particion_val,
        )

        # ─── 3. Registrar dataset en BD ───
        logger.info("═══ PASO 3/10: Registrando dataset en BD ═══")
        session = database.get_session()
        dataset_record = DatasetWithdrawalPrediction(
            start_date=df["fecha_transaccion"].min().date(),
            end_date=df["fecha_transaccion"].max().date(),
            count_total=len(df),
            count_train=len(data.train.df),
            count_test=len(data.test.df),
            features=FEATURES,
            target=TARGET,
        )
        session.add(dataset_record)
        session.commit()
        logger.info(f"📝 Dataset registrado con ID: {dataset_record.id}")

        # ─── 4. Analisis de Distribucion de datos ───
        logger.info("═══ PASO 4/10: Analisis de Distribucion de Datos (PSI) ═══")
        data_distribution = obtener_distribucion_actual_atm_features()
        psi = get_psi(data_distribution)
        logger.info(f"PSI: calculado exitosamente")

        # ─── 5. Optimizar hiperparámetros ───
        logger.info("═══ PASO 5/10: Optimización Optuna ═══")
        study = ModelOptimizer.optimizar_hiperparametros(
            X_train=data.train.X,
            y_train_log=data.train.y_log,
            n_trials=request.optuna_trials,
        )
        best_params = study.best_params

        # ─── 7. Entrenar modelo final ───
        logger.info("═══ PASO 7/10: Entrenando modelo final ═══")
        new_model = ModelOptimizer.entrenar_modelo_final(
            best_params=best_params,
            X_train=data.train_final.X,
            y_train_log=data.train_final.y_log,
            X_val=data.val.X,
            y_val_log=data.val.y_log,
            features=FEATURES,
        )

        # ─── 7. Evaluar nuevo modelo ───
        logger.info("═══ PASO 7/10: Evaluando challenger ═══")
        training_time = time.time() - start_time
        metrics_beta = ModelEvaluator.evaluar_modelo(
            new_model, data.test.X, data.test.y,
        )
        importancias = ModelEvaluator.obtener_importancia_features(new_model, FEATURES)
        ic = ModelEvaluator.calcular_intervalo_confianza(
            data.test.X, data.test.y, new_model,
        )

        challenger_metrics = TrainingMetrics(
            mae=metrics_beta["mae"],
            mape=metrics_beta["mape"],
            rmse=metrics_beta["rmse"],
            training_time_sec=training_time,
        )

        # ─── 8. Descargar y evaluar champion ───
        logger.info("═══ PASO 8/10: Evaluando champion actual ═══")
        print("═══ PASO 8/10: Evaluando champion actual ═══")
        provider = AtmModelProvider()
        provider.init_dagshub_connection()
        champion_model = provider.obtener_modelo_produccion(force_download=True)

        champion_metrics = None
        metrics_prod = None
        if champion_model is not None:
            try:
                metrics_prod = ModelEvaluator.evaluar_modelo(
                    champion_model, data.test.X, data.test.y,
                )
                champion_metrics = TrainingMetrics(
                    mae=metrics_prod["mae"],
                    mape=metrics_prod["mape"],
                    rmse=metrics_prod["rmse"],
                    training_time_sec=0,
                )
            except Exception as e:
                logger.warning(f"⚠️ No se pudo evaluar champion: {e}")

        # ─── 9. Decidir promoción ───
        logger.info("═══ PASO 9/10: Decidiendo promoción ═══")
        deployment_status = "KEEP_CHAMPION"
        version_tag = None
        margin_improvement = None
        dagshub_verified = False

        should_promote = False
        if metrics_prod is not None:
            should_promote = ModelEvaluator.evaluar_cambio_significativo(
                mape_prod=metrics_prod["mape"],
                mape_beta=metrics_beta["mape"],
                tolerancia=request.tolerancia_mape,
            )
            margin_improvement = (
                (metrics_prod["mape"] - metrics_beta["mape"]) / metrics_prod["mape"]
            )
        else:
            # Cold start: no hay champion, promover automáticamente
            should_promote = True
            logger.info("🆕 Cold start: no hay champion, promoviendo nuevo modelo")

        # Registrar auditoría independiente a si se promueve o no
        audit = _registrar_self_training_audit_withdrawal_model(
            session=session,
            new_model=new_model,
            metrics_beta=metrics_beta,
            best_params=best_params,
            margin_improvement=margin_improvement,
            start_datetime=start_datetime,
            dataset_id=dataset_record.id, # type: ignore
            psi=psi,
        )

        if should_promote:
            deployment_status = _promover_modelo(
                session = session ,
                new_model = new_model,
                provider = provider,
                audit = audit,
                importancias = importancias,
                ic= ic,
            )

            if deployment_status == "NEW_CHAMPION":
                ultima_version = audit.model_name
                version_tag = f"v{ultima_version}"
                dagshub_verified = AtmModelProvider.verificar_integridad(version_tag)

        # ─── 10. Log MLflow ───
        logger.info("═══ PASO 10/10: Logging MLflow ═══")
        _log_mlflow(
            challenger_metrics, champion_metrics, best_params,
            study.best_value, deployment_status,
        )

        return TrainingResponse(
            metrics_challenger=challenger_metrics,
            metrics_champion=champion_metrics,
            best_params=best_params,
            n_trials=request.optuna_trials,
            total_samples=len(df),
            train_samples=len(data.train.df),
            test_samples=len(data.test.df),
            feature_importances=importancias,
            confidence_interval=ConfidenceInterval(**ic) if ic else None,
            deployment_status=deployment_status,
            version_tag=version_tag,
            margin_improvement=margin_improvement,
            dagshub_verified=dagshub_verified,
        )

    except Exception as e:
        logger.error(f"❌ Error en autoentrenamiento: {e}", exc_info=True)
        if session:
            session.rollback()
        raise
    finally:
        if session:
            session.close()
            logger.info("🔒 Sesión de BD cerrada")


# ═══════════════════════════════════════════════════════════
# FUNCIONES INTERNAS
# ═══════════════════════════════════════════════════════════
def _registrar_self_training_audit_withdrawal_model (
    session,
    new_model,
    metrics_beta: dict,
    best_params: dict,
    margin_improvement,
    start_datetime: datetime,
    dataset_id: int,
    psi: dict,
) -> SelfTrainingAuditWithdrawalModel:
    try:
        # Obtener versión
        name_model = type(new_model).__name__ or "ATM model prediction"

        ultima_version = consultar_ultima_version_modelo(name_model)
        version_tag = f"{name_model}_v{ultima_version + 1}"

        # Obtener modelo actual
        model_rival = (
            session.query(SelfTrainingAuditWithdrawalModel)
            .filter(SelfTrainingAuditWithdrawalModel.is_production == True)
            .first()
        )

        # Guardar auditoría
        audit = SelfTrainingAuditWithdrawalModel(
            model_name=version_tag,
            mae=metrics_beta["mae"],
            mape=metrics_beta["mape"],
            rmse=metrics_beta["rmse"],
            margin_improvement=margin_improvement or 0,
            training_duration_minutes=int(
                (datetime.now() - start_datetime).total_seconds() / 60
            ),
            start_training=start_datetime,
            end_training=datetime.now(),
            hyperparameters=best_params,
            is_production=False,
            compared_to_model=model_rival.id if model_rival else None,
            id_dataset_withdrawal_prediction=dataset_id,
            psi_baseline=psi,
        )
        session.add(audit)
        session.commit()
        logger.info(f"📝 Auditoría guardada: {version_tag} (ID: {audit.id})")
        return audit
    except Exception as e:
        logger.error(f"❌ Error al registrar auditoría: {e}")
        raise

def _promover_modelo(
    session,
    new_model,
    provider: AtmModelProvider,
    audit: SelfTrainingAuditWithdrawalModel,
    importancias: dict,
    ic: dict
) -> str:
    """
    Ejecuta la promoción del nuevo modelo:
    1. Desactiva modelo anterior
    2. Activa nuevo modelo
    3. Sube a DagsHub
    4. Notifica backend Java

    Returns:
        "NEW_CHAMPION" o "UPLOAD_FAILED".
    """
    try:
        # Desactivar modelo anterior
        modelo_actual = (
            session.query(WithdrawalModel)
            .filter(WithdrawalModel.is_active == True)
            .first()
        )
        if modelo_actual:
            modelo_actual.is_active = False
            modelo_actual.end_date = date.today()
            session.query(SelfTrainingAuditWithdrawalModel).filter(
                SelfTrainingAuditWithdrawalModel.id
                == modelo_actual.id_self_training_audit_withdrawal_model
            ).update({"is_production": False})
            logger.info(f"🔄 Modelo rival (ID: {modelo_actual.id}) desactivado")

        # Activar nuevo modelo
        new_production = WithdrawalModel(
            confidence_level=ic.get("confidence_level", 95),
            end_date=None,
            is_active=True,
            margin=ic["margin_error"],
            sigma=ic["sigma"],
            start_date=date.today(),
            t_crit=ic["t_crit"],
            id_self_training_audit_withdrawal_model=audit.id,
            importances_features=importancias,
        )
        session.add(new_production)

        audit.is_production = True # type: ignore
        session.add(audit)
        session.commit()
        logger.info(f"✅ Nuevo modelo activado (ID: {new_production.id})")

        # Subir a DagsHub
        uploaded = provider.actualizar_modelo_produccion(new_model, audit.model_name) # type: ignore
        if not uploaded:
            logger.error("❌ Fallo al subir a DagsHub")
            return "UPLOAD_FAILED"

        # Notificar backend Java
        _notificar_backend_java()

        return "NEW_CHAMPION"

    except Exception as e:
        session.rollback()
        logger.error(f"❌ Error en promoción: {e}", exc_info=True)
        return "UPLOAD_FAILED"


def _notificar_backend_java():
    """Notifica al backend Java que hay un nuevo modelo disponible."""
    try:
        response = requests.post(UPDATE_MODEL_API_URL, timeout=10) # type: ignore
        response.raise_for_status()
        logger.info(f"📡 Backend Java notificado: {response.json()}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ No se pudo notificar al backend Java: {e}")


def _log_mlflow(
    challenger_metrics: TrainingMetrics,
    champion_metrics,
    best_params: dict,
    best_cv_value: float,
    deployment_status: str,
):
    """Registra métricas y parámetros en MLflow."""
    try:
        with mlflow.start_run(run_name="retiro_atm_self_training"):
            # Parámetros
            mlflow.log_params(best_params)
            mlflow.log_param("deployment_status", deployment_status)

            # Métricas del challenger
            mlflow.log_metric("challenger_mae", challenger_metrics.mae)
            mlflow.log_metric("challenger_mape", challenger_metrics.mape)
            mlflow.log_metric("challenger_rmse", challenger_metrics.rmse)
            mlflow.log_metric("best_cv_mape", best_cv_value)
            mlflow.log_metric("training_time_sec", challenger_metrics.training_time_sec)

            # Métricas del champion (si existen)
            if champion_metrics:
                mlflow.log_metric("champion_mae", champion_metrics.mae)
                mlflow.log_metric("champion_mape", champion_metrics.mape)
                mlflow.log_metric("champion_rmse", champion_metrics.rmse)

        logger.info("📊 MLflow run registrado")
    except Exception as e:
        logger.warning(f"⚠️ Error al registrar en MLflow: {e}")
