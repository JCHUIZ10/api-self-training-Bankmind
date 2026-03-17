import logging
import sys
from .repository import db_queries as repo
from .service import (
    ejecutar_pipeline_features,
    recuperar_predicciones_faltantes,
    calcular_metricas,
    obtener_veredicto_error,
    calcular_psi,
    evaluar_alertas_psi,
    generar_veredicto_final,
)

from retiro_atm.self_train.training_service import ejecutar_autoentrenamiento
from retiro_atm.schemas import TrainingRequest

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# CICLO DE MONITOREO
# ══════════════════════════════════════════════════════════════════

def ejecutar_monitoreo(engine) -> None:
    """
    Un ciclo completo de monitoreo:
      1. Features ATM
      2. Predicciones faltantes
      3. Métricas de error
      4. PSI
      5. Veredicto final + persistencia
    """
    try:
        logger.info("═" * 60)
        logger.info("INICIO DEL CICLO DE MONITOREO")
        logger.info("═" * 60)

        # ── 1. Pipeline de features ──────────────────────────────
        logger.info("[ 1/5 ] Calculando features ATM...")
        ejecutar_pipeline_features(engine)

        # ── 2. Modelo activo ─────────────────────────────────────
        logger.info("[ 2/5 ] Cargando modelo activo y comparativa real vs. predicción...")
        df_modelo = repo.obtener_modelo_activo(engine)
        if df_modelo.empty:
            raise ValueError("No se encontró ningún modelo activo en la base de datos.")

        model_id = int(df_modelo["id"].iloc[0])
        margen   = float(df_modelo["margin"].squeeze()) # type: ignore
        logger.info("Modelo activo: id=%d, margen=%.2f", model_id, margen)

        df_real_vs_pred = repo.obtener_real_vs_prediccion(engine, model_id)

        # ── 3. Predicciones faltantes ────────────────────────────
        logger.info("[ 3/5 ] Verificando predicciones faltantes...")
        df_real_vs_pred = recuperar_predicciones_faltantes(
            engine, df_real_vs_pred, model_id, margen
        )

        # ── 4. Métricas de error ─────────────────────────────────
        logger.info("[ 4/5 ] Calculando métricas de error...")
        mae, rmse, mape = calcular_metricas(
            df_real_vs_pred["amount"],
            df_real_vs_pred["predicted_value"],
        )
        error_veredicto, icono = obtener_veredicto_error(mape)
        logger.info(
            f"{icono} Métricas — MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% → {error_veredicto}"
        )

        # ── 5. PSI ───────────────────────────────────────────────
        logger.info("[ 5/5 ] Calculando PSI...")
        baseline    = repo.obtener_psi_baseline(engine)
        df_psi_prod = repo.obtener_datos_psi_actual(engine)
        psi_results = calcular_psi(baseline, df_psi_prod)
        psi_eval    = evaluar_alertas_psi(psi_results)
        logger.info("PSI → Decisión: %s | %s", psi_eval["decision"], psi_eval["message"])

        # ── Veredicto final ──────────────────────────────────────
        reentrenar = generar_veredicto_final(psi_eval["decision"], error_veredicto)
        logger.info("Reentrenamiento requerido: %s", reentrenar)

        # ── Persistir resultado ──────────────────────────────────
        repo.insertar_resultado_monitoreo(
            engine      = engine,
            model_id    = model_id,
            mae         = mae,
            rmse        = rmse,
            mape        = mape,
            psi_results = psi_results,
            psi_eval    = psi_eval,
            reentrenar  = reentrenar,
        )
        logger.info("Resultado de monitoreo persistido correctamente.")
        logger.info("CICLO COMPLETADO ✓")

        if reentrenar:
            logger.info("Autoentrenamiento Lanzado")
            config = TrainingRequest()
            ejecutar_autoentrenamiento(config)

    except Exception:
        logger.exception("Error crítico en el ciclo de monitoreo")
        raise
    finally:
        engine.dispose()