import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from psycopg2.extras import execute_batch
from sqlalchemy import text, types
from sqlalchemy.engine import Engine

from retiro_atm.monitoring.model.schemas import InputDataRetiroAtm

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# TRANSACCIONES
# ══════════════════════════════════════════════════════════════════

def obtener_ultima_fecha_sincronizacion(engine: Engine) -> Optional[pd.Timestamp]:
    """Retorna la fecha mínima de transacción del último sync."""
    sql = text("""
        SELECT MIN(dat.transaction_date)
        FROM daily_atm_transactions dat
        WHERE dat.id_sync = (SELECT MAX(sl.id_sync) FROM sync_logs sl)
    """)
    with engine.connect() as conn:
        return conn.execute(sql).scalar()


def obtener_transacciones(engine: Engine) -> pd.DataFrame:
    """
    Carga las transacciones de retiro a partir de 11 días antes
    de la última sincronización.
    """
    ultima_fecha = obtener_ultima_fecha_sincronizacion(engine)
    if ultima_fecha is None:
        raise ValueError("No se encontró ninguna sincronización registrada.")

    fecha_minima = pd.Timestamp(ultima_fecha) - pd.Timedelta(days=11)
    logger.info("Cargando transacciones desde %s", fecha_minima.date())

    sql = """
        SELECT
            dat.id_transaction,
            dat.transaction_date,
            dat.id_atm,
            dat.amount
        FROM daily_atm_transactions dat
        INNER JOIN atms a ON dat.id_atm = a.id_atm
        WHERE dat.transaction_date  >= %(fecha_minima)s
          AND dat.transaction_type  = 'WITHDRAWAL'
          AND a.active
        ORDER BY dat.id_atm, dat.transaction_date
    """
    df = pd.read_sql(sql, engine, params={"fecha_minima": fecha_minima})
    # Forzar tipo datetime (defensivo ante distintos drivers)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    logger.info("Transacciones cargadas: %d filas", len(df))
    return df


# ══════════════════════════════════════════════════════════════════
# FEATURES ATM
# ══════════════════════════════════════════════════════════════════

def insertar_atm_features(engine: Engine, df_final: pd.DataFrame) -> int:
    """
    Inserta el DataFrame de features en la tabla atm_features.
    Retorna el número de registros insertados.
    Usa una transacción explícita: si falla, hace rollback automático.
    """
    with engine.begin() as conn:           # begin() hace commit/rollback automático
        df_final.to_sql(
            name="atm_features",
            con=conn,
            schema="public",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
            dtype={"dynamic_features": types.JSON},
        )
    return len(df_final)


def obtener_datos_psi_actual(engine: Engine) -> pd.DataFrame:
    """Carga las features de los últimos 30 días para calcular PSI actual."""
    sql = """
        SELECT
            (af.dynamic_features->>'lag1')::float                  AS lag1,
            (af.dynamic_features->>'lag5')::float                  AS lag5,
            (af.dynamic_features->>'lag11')::float                 AS lag11,
            (af.dynamic_features->>'tendencia_lags')::float        AS tendencia_lags,
            (af.dynamic_features->>'ratio_finde_vs_semana')::float AS ratio_finde_vs_semana,
            (af.dynamic_features->>'retiros_finde_anterior')::float AS retiros_finde_anterior,
            (af.dynamic_features->>'retiros_domingo_anterior')::float AS retiros_domingo_anterior
        FROM atm_features af
        WHERE af.reference_date >= (
            SELECT MAX(reference_date) - INTERVAL '30 days'
            FROM atm_features
        )
    """
    df = pd.read_sql(sql, engine)
    # Descartar filas con nulos en las primeras 7 features
    return df.dropna(subset=df.columns[:7])


# ══════════════════════════════════════════════════════════════════
# MODELOS Y PREDICCIONES
# ══════════════════════════════════════════════════════════════════

def obtener_modelo_activo(engine: Engine) -> pd.DataFrame:
    """Retorna id y margen del modelo de retiro activo."""
    sql = "SELECT wm.id, wm.margin FROM withdrawal_models wm WHERE wm.is_active = true LIMIT 1"
    return pd.read_sql(sql, engine)


def obtener_real_vs_prediccion(engine: Engine, model_id: int) -> pd.DataFrame:
    """Compara retiros reales contra las predicciones del modelo activo."""
    sql = text("""
        SELECT
            dat.id_atm,
            dat.id_transaction,
            dat.transaction_date,
            dat.amount,
            dwp.predicted_value
        FROM daily_atm_transactions dat
        LEFT JOIN daily_withdrawal_prediction dwp
            ON  dwp.prediction_date      = dat.transaction_date
            AND dwp.id_atm               = dat.id_atm
            AND dwp.id_withdrawal_model  = :model_id
        WHERE dat.id_sync         = (SELECT MAX(sl.id_sync) FROM sync_logs sl)
          AND dat.transaction_type = 'WITHDRAWAL'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, con=conn, params={"model_id": model_id})
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def obtener_datos_faltantes(
    conn,                       # conexión psycopg2 cruda
    lista_ids: tuple,
) -> list[InputDataRetiroAtm]:
    """
    Obtiene las features de transacciones sin predicción para
    re-enviarlas a la API.
    """
    sql = """
        SELECT
            a.id_atm                                                 AS atm,
            dat.transaction_date                                     AS prediction_date,
            af.day_of_week                                           AS "diaSemana",
            a.id_location_type                                       AS ubicacion,
            (af.dynamic_features->>'lag1')::float                   AS lag1,
            (af.dynamic_features->>'lag5')::float                   AS lag5,
            (af.dynamic_features->>'lag11')::float                  AS lag11,
            (af.dynamic_features->>'tendencia_lags')::float         AS tendencia_lags,
            (af.dynamic_features->>'ratio_finde_vs_semana')::float  AS ratio_finde_vs_semana,
            (af.dynamic_features->>'retiros_finde_anterior')::float AS retiros_finde_anterior,
            (af.dynamic_features->>'retiros_domingo_anterior')::float AS retiros_domingo_anterior,
            (af.dynamic_features->>'domingo_bajo')::int             AS domingo_bajo,
            (af.dynamic_features->>'caida_reciente')::int           AS caida_reciente,
            w.impact                                                 AS ambiente
        FROM atm_features af
        INNER JOIN daily_atm_transactions dat ON af.id_transaction   = dat.id_transaction
        INNER JOIN atms a                      ON dat.id_atm          = a.id_atm
        INNER JOIN weathers w                  ON dat.id_weather       = w.id_weather
        WHERE af.id_transaction IN %s
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, (lista_ids,))
        cols    = [desc[0] for desc in cursor.description]
        results = [dict(zip(cols, row)) for row in cursor.fetchall()]

    return [InputDataRetiroAtm(**row) for row in results]


def insertar_predicciones(conn, lista_predicciones: list, model_id: int, margen: float) -> None:
    """
    Inserta las predicciones faltantes en daily_withdrawal_prediction.
    Usa execute_batch para rendimiento. La transacción (commit/rollback)
    la controla el llamador.
    """
    import pandas as pd

    df = pd.DataFrame(lista_predicciones)
    ahora = datetime.now()

    df["lower_bound"]         = df["retiro"].astype(float) - margen
    df["upper_bound"]         = df["retiro"].astype(float) + margen
    df["id_withdrawal_model"] = model_id
    df["registration_date"]   = ahora

    data = df[[
        "lower_bound", "retiro", "prediction_date",
        "upper_bound", "atm", "id_withdrawal_model", "registration_date",
    ]].values.tolist()

    sql = """
        INSERT INTO public.daily_withdrawal_prediction
            (lower_bound, predicted_value, prediction_date, upper_bound,
             id_atm, id_withdrawal_model, registration_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cursor:
        execute_batch(cursor, sql, data)


# ══════════════════════════════════════════════════════════════════
# PSI BASELINE
# ══════════════════════════════════════════════════════════════════

def obtener_psi_baseline(engine: Engine) -> dict:
    """
    Carga el baseline PSI del modelo en producción.
    El campo psi_baseline se almacena como JSON en la DB.
    """
    sql = text("""
        SELECT stawm.psi_baseline
        FROM self_training_audit_withdrawal_model stawm
        WHERE stawm.is_production
    """)
    with engine.connect() as conn:
        raw = conn.execute(sql).scalar()

    if raw is None:
        raise ValueError("No se encontró baseline PSI para el modelo en producción.")

    # El campo puede venir como str (JSON) o ya como dict (según driver)
    return raw if isinstance(raw, dict) else json.loads(raw)


# ══════════════════════════════════════════════════════════════════
# MONITOREO
# ══════════════════════════════════════════════════════════════════

def insertar_resultado_monitoreo(
    engine:      Engine,
    model_id:    int,
    mae:         float,
    rmse:        float,
    mape:        float,
    psi_results: dict,
    psi_eval:    dict,
    reentrenar:  bool,
) -> None:
    """
    Persiste el resultado completo del ciclo de monitoreo.
    Encapsulado en una transacción: si falla, hace rollback.
    """
    sql = text("""
        INSERT INTO public.performance_monitor_model_atm
            (action, created_at, decision, detail, mae, mape,
             message, psi_results, rmse, summary,
             id_withdrawal_model, need_selftraining)
        VALUES
            (:accion, NOW(), :decision, :detalles, :mae, :mape,
             :mensaje, :psi_results, :rmse, :resumen,
             :id_model, :need)
    """)
    params = {
        "accion":      psi_eval["action"],
        "decision":    psi_eval["decision"],
        "detalles":    json.dumps(psi_eval["detail"],   ensure_ascii=False),
        "mae":         float(mae),
        "mape":        float(mape),
        "mensaje":     psi_eval["message"],
        "psi_results": json.dumps(psi_results,          ensure_ascii=False),
        "rmse":        float(rmse),
        "resumen":     json.dumps(psi_eval["summary"],  ensure_ascii=False),
        "id_model":    model_id,
        "need":        reentrenar,
    }
    with engine.begin() as conn:
        conn.execute(sql, params)
