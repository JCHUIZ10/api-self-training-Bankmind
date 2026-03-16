import logging
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sqlalchemy.engine import Engine

from retiro_atm.monitoring.config import PREDICTION_API_URL, PSI_CRITICAL_THRESHOLD, PSI_WARNING_THRESHOLD
from retiro_atm.monitoring.config import PSI_CRITICAL_PCT_FORCE, PSI_WARNING_PCT_ALARM
from retiro_atm.monitoring.service.atm_feature_generator import AtmFeatureGenerator
from retiro_atm.monitoring.repository import db_queries as repo

logger = logging.getLogger(__name__)

# Columnas que irán al modelo
FEATURE_COLS = [
    "tendencia_lags", "lag_1", "lag_5", "lag_11",
    "caida_reciente", "retiros_finde_anterior",
    "retiros_domingo_anterior", "ratio_finde_vs_semana",
    "domingo_bajo",
]
META_COLS = ["id_atm", "dia_mes", "mes", "diaSemana", "transaction_date"]



# ══════════════════════════════════════════════════════════════════
# 1. PIPELINE DE FEATURES
# ══════════════════════════════════════════════════════════════════

def ejecutar_pipeline_features(engine: Engine) -> pd.DataFrame:
    """
    Carga transacciones → calcula features → persiste en atm_features.
    Retorna el DataFrame listo para métricas.
    """
    df_tx = repo.obtener_transacciones(engine)

    generator = AtmFeatureGenerator(df_tx)
    df_enriquecido = generator.calcular_features()

    df_features = df_enriquecido[FEATURE_COLS + ["amount"] + META_COLS + ["id_transaction"]].copy()

    # Serializar features dinámicas a JSON por fila
    df_features["dynamic_features"] = df_features[FEATURE_COLS].apply(
        lambda x: {
            "lag1":                     x["lag_1"],
            "lag5":                     x["lag_5"],
            "lag11":                    x["lag_11"],
            "domingo_bajo":             int(x["domingo_bajo"]),
            "caida_reciente":           int(x["caida_reciente"]),
            "tendencia_lags":           float(x["tendencia_lags"]),
            "ratio_finde_vs_semana":    float(x["ratio_finde_vs_semana"]),
            "retiros_finde_anterior":   float(x["retiros_finde_anterior"]),
            "retiros_domingo_anterior": float(x["retiros_domingo_anterior"]),
        },
        axis=1,
    )

    # Mapear columnas al esquema SQL
    df_final = pd.DataFrame({
        "day_of_month":          df_features["dia_mes"],
        "day_of_week":           df_features["diaSemana"],
        "month":                 df_features["mes"],
        "reference_date":        df_features["transaction_date"],
        "withdrawal_amount_day": df_features["amount"],
        "id_transaction":        df_features["id_transaction"],
        "dynamic_features":      df_features["dynamic_features"],
        "created_at":            datetime.now(),
    })

    insertados = repo.insertar_atm_features(engine, df_final)
    logger.info("Features insertadas: %d registros", insertados)

    return df_enriquecido


# ══════════════════════════════════════════════════════════════════
# 2. RECUPERACIÓN DE PREDICCIONES FALTANTES
# ══════════════════════════════════════════════════════════════════

def recuperar_predicciones_faltantes(
    engine: Engine,
    df_real_vs_pred: pd.DataFrame,
    model_id: int,
    margen: float,
) -> pd.DataFrame:
    """
    Detecta transacciones sin predicción y las envía a la API.
    Retorna el DataFrame actualizado con las nuevas predicciones.
    """

    # Asegurar que la columna exista
    if "predicted_value" not in df_real_vs_pred.columns:
        df_real_vs_pred["predicted_value"] = None

    faltantes = df_real_vs_pred[df_real_vs_pred["predicted_value"].isnull()]

    if faltantes.empty:
        logger.info("No hay predicciones faltantes.")
        return df_real_vs_pred

    ids_faltantes = tuple(faltantes["id_transaction"].tolist())
    logger.info("Predicciones faltantes: %d transacciones", len(ids_faltantes))

    raw_conn = engine.raw_connection()

    try:
        datos = repo.obtener_datos_faltantes(raw_conn, ids_faltantes)
        payload = [item.model_dump(mode="json") for item in datos]

        response = requests.post(PREDICTION_API_URL, json=payload, timeout=30)
        response.raise_for_status()

        predicciones = response.json()
        logger.info("API respondió %d predicciones", len(predicciones))

        # Guardar en BD
        repo.insertar_predicciones(raw_conn, predicciones, model_id, margen)
        raw_conn.commit()

        # Convertir predicciones a DataFrame
        df_nuevas = pd.DataFrame(predicciones).rename(columns={
            "atm": "id_atm",
            "prediction_date": "transaction_date",
            "retiro": "predicted_value",
        })

        df_nuevas["transaction_date"] = pd.to_datetime(
            df_nuevas["transaction_date"]
        )

        # Crear mapa para actualización rápida
        mapa_predicciones = {
            (row.id_atm, row.transaction_date): row.predicted_value
            for row in df_nuevas.itertuples()
        }

        # Actualizar solo valores faltantes
        mask = df_real_vs_pred["predicted_value"].isna()

        df_real_vs_pred.loc[mask, "predicted_value"] = (
            df_real_vs_pred.loc[mask]
            .apply(
                lambda r: mapa_predicciones.get(
                    (r["id_atm"], r["transaction_date"])
                ),
                axis=1
            )
        )

    except requests.exceptions.HTTPError as e:
        raw_conn.rollback()
        logger.error("Error HTTP en API de predicción: %s", e.response.text)

    except Exception as e:
        raw_conn.rollback()
        logger.exception("Error al recuperar predicciones: %s", e)

    finally:
        raw_conn.close()

    return df_real_vs_pred

# ══════════════════════════════════════════════════════════════════
# 3. MÉTRICAS DE ERROR
# ══════════════════════════════════════════════════════════════════

def calcular_metricas(real: pd.Series, prediccion: pd.Series) -> tuple[float, float, float]:
    """Calcula MAE, RMSE y MAPE. Ignora filas con NaN."""
    mask = prediccion.notna() & real.notna()
    r = real[mask].to_numpy()
    p = prediccion[mask].to_numpy()

    mae  = float(np.mean(np.abs(r - p)))
    rmse = float(np.sqrt(np.mean((r - p) ** 2)))
    mape = float(np.mean(np.abs((r - p) / (r + 1e-9))) * 100)
    return mae, rmse, mape


def obtener_veredicto_error(mape: float) -> tuple[str, str]:
    """
    Clasifica la calidad de predicción según el MAPE.
    Returns: (veredicto, icono)
    """
    # Normalizar: si viene en decimal (0.05 → 5 %)
    mape_pct = mape * 100 if mape < 1 else mape

    if mape_pct <= 10:
        return "OK",      "🟢"
    elif mape_pct <= 25:
        return "ALERTA",  "🟡"
    else:
        return "CRITICO", "🔴"


# ══════════════════════════════════════════════════════════════════
# 4. PSI — POPULATION STABILITY INDEX
# ══════════════════════════════════════════════════════════════════

def calcular_psi(baseline_metadata: dict, df_produccion: pd.DataFrame, epsilon: float = 1e-4) -> dict:
    """
    Calcula el PSI por feature comparando distribución baseline vs. producción.
    """
    psi_results = {}

    for feature, meta in baseline_metadata.items():

        # Feature sin bins válidos en el baseline
        if meta.get("bins") is None:
            psi_results[feature] = {
                "psi":    None,
                "alert":  "skipped",
                "reason": meta.get("warning", "unknown"),
            }
            continue

        # Feature ausente en producción
        if feature not in df_produccion.columns:
            psi_results[feature] = {
                "psi":    None,
                "alert":  "skipped",
                "reason": "feature ausente en producción",
            }
            continue

        # Restaurar infinitos en los extremos del bin
        bins = meta["bins"].copy()
        bins[0]  = -np.inf
        bins[-1] =  np.inf
        bins = np.array(bins, dtype=float)

        expected_pct = np.array(meta["expected_pct"])
        serie_prod   = df_produccion[feature].dropna()
        counts, _    = np.histogram(serie_prod, bins=bins)
        total        = counts.sum()

        if total == 0:
            psi_results[feature] = {
                "psi":    None,
                "alert":  "skipped",
                "reason": "sin datos válidos en producción",
            }
            continue

        actual_pct = counts / total
        psi_value  = float(np.sum(
            (actual_pct - expected_pct)
            * np.log((actual_pct + epsilon) / (expected_pct + epsilon))
        ))

        if psi_value < PSI_WARNING_THRESHOLD:
            alert = "OK"
        elif psi_value < PSI_CRITICAL_THRESHOLD:
            alert = "WARNING"
        else:
            alert = "CRITICAL"

        psi_results[feature] = {
            "psi":          round(psi_value, 4),
            "alert":        alert,
            "actual_pct":   actual_pct.tolist(),
            "expected_pct": expected_pct.tolist(),
            "prod_samples": int(total),
            "prod_null_pct": float(df_produccion[feature].isna().mean()),
        }

    return psi_results


def evaluar_alertas_psi(psi_results: dict) -> dict:
    """
    Determina el estado global del modelo a partir del reporte PSI.
    """
    valid    = {f: r for f, r in psi_results.items() if r.get("psi") is not None}
    skipped  = {f: r for f, r in psi_results.items() if r.get("psi") is None}

    if not valid:
        return {
            "decision": "INSUFICIENTE",
            "message":  "No hay features válidas para evaluar PSI.",
            "action":   "Acumular más datos antes de evaluar.",
            "summary":  {},
            "detail":   {},
            "skipped":  list(skipped),
        }

    total     = len(valid)
    critical  = {f: r for f, r in valid.items() if r["psi"] >= PSI_CRITICAL_THRESHOLD}
    warning   = {f: r for f, r in valid.items() if PSI_WARNING_THRESHOLD <= r["psi"] < PSI_CRITICAL_THRESHOLD}
    ok        = {f: r for f, r in valid.items() if r["psi"] < PSI_WARNING_THRESHOLD}

    pct_crit  = len(critical) / total
    pct_warn_plus = (len(critical) + len(warning)) / total

    worst_f   = max(valid, key=lambda f: valid[f]["psi"])
    worst_psi = valid[worst_f]["psi"]

    # ── Decisión global ──────────────────────────────────────────
    if pct_crit >= PSI_CRITICAL_PCT_FORCE:
        decision = "REENTRENAMIENTO_OBLIGATORIO"
        message  = f"{len(critical)}/{total} features en CRITICAL ({pct_crit:.0%})."
        action   = "Reentrenar el modelo de forma inmediata."

    elif worst_psi >= 0.35:
        decision = "REENTRENAMIENTO_OBLIGATORIO"
        message  = f"Feature '{worst_f}' tiene PSI={worst_psi} — deriva extrema detectada."
        action   = "Reentrenar el modelo de forma inmediata."

    elif pct_warn_plus >= PSI_WARNING_PCT_ALARM:
        decision = "ALARMA"
        message  = f"{len(critical)} CRITICAL y {len(warning)} WARNING de {total} features ({pct_warn_plus:.0%})."
        action   = "Monitorear diariamente. Preparar reentrenamiento si la tendencia continúa."

    elif critical:
        decision = "ALARMA"
        message  = f"{len(critical)} feature(s) en CRITICAL: {list(critical)}."
        action   = "Investigar las features afectadas. Evaluar reentrenamiento."

    else:
        decision = "OK"
        message  = f"Todas las features ({total}) dentro de umbrales normales."
        action   = "Ninguna acción requerida."

    return {
        "decision": decision,
        "message":  message,
        "action":   action,
        "summary": {
            "total_features":   total,
            "n_critical":       len(critical),
            "n_warning":        len(warning),
            "n_ok":             len(ok),
            "n_skipped":        len(skipped),
            "pct_critical":     round(pct_crit, 4),
            "pct_warning_plus": round(pct_warn_plus, 4),
            "worst_feature":    worst_f,
            "worst_psi":        worst_psi,
            "evaluated_at":     pd.Timestamp.now().isoformat(),
        },
        "detail": {
            "CRITICAL": {f: r["psi"] for f, r in critical.items()},
            "WARNING":  {f: r["psi"] for f, r in warning.items()},
            "OK":       {f: r["psi"] for f, r in ok.items()},
            "SKIPPED":  {f: r.get("reason", "") for f, r in skipped.items()},
        },
    }


# ══════════════════════════════════════════════════════════════════
# 5. VEREDICTO FINAL (PSI + Error)
# ══════════════════════════════════════════════════════════════════

def generar_veredicto_final(psi_decision: str, error_decision: str) -> bool:
    """
    Combina los veredictos de estabilidad (PSI) y precisión (error).
    Returns: True si se requiere reentrenamiento.
    """
    if psi_decision == "REENTRENAMIENTO_OBLIGATORIO":
        return True
    if error_decision == "CRITICO":
        return True
    # Doble amarillo → prevención
    if psi_decision == "ALARMA" and error_decision == "ALERTA":
        return True
    return False
