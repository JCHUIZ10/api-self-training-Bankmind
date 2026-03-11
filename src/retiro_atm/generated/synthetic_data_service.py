"""
Script de sincronización de transacciones ATM para BankMind.

"""

import json
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from retiro_atm.monitoring.orquestador_monitoreo import ejecutar_monitoreo

from retiro_atm import database

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clase de auditoría: construye el process_log JSONB paso a paso
# ---------------------------------------------------------------------------
class ProcessAudit:
    """
    Acumula entradas de auditoría y las serializa como lista JSON.
    Cada entrada tiene la forma:
        {
            "timestamp": "2026-01-31T10:00:01.123",
            "action":    "NOMBRE_ACCION",
            "status":    "OK" | "ERROR" | "INFO",
            "details":   { ... métricas del paso ... }
        }
    """

    def __init__(self):
        self._entries: list[dict] = []

    def log(self, action: str, status: str = "OK", **details):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "action":    action,
            "status":    status,
            "details":   details if details else {}
        }
        self._entries.append(entry)
        log.info("[AUDIT] %s → %s | %s", action, status, details)

    def to_json_array(self) -> list:
        """Devuelve la lista lista para guardar en JSONB."""
        return self._entries

    def to_json_string(self) -> str:
        return json.dumps(self._entries, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# 1. Obtener estado actual de ATMs
# ---------------------------------------------------------------------------
def obtener_ultimo_estado_atm(engine) -> pd.DataFrame:
    query = """
    SELECT
        acs.current_balance,
        acs.id_atm,
        acs.last_deposit_date,
        acs.last_reload_date,
        acs.last_transaction_date,
        acs.last_withdrawal_date,
        acs.last_sync_id
    FROM public.atm_current_status acs
    INNER JOIN public.atms a ON a.id_atm = acs.id_atm
    WHERE a.active = true;
    """
    return pd.read_sql(query, engine)


# ---------------------------------------------------------------------------
# 2. Asignar impacto climático a cada transacción
# ---------------------------------------------------------------------------
def buscar_impacto_climatico(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fechas_unicas = df["transaction_date"].unique()
    clima_por_fecha = {f: np.random.randint(1, 4) for f in fechas_unicas}
    df["id_weather"] = df["transaction_date"].map(clima_por_fecha)
    return df


# ---------------------------------------------------------------------------
# 3. Corregir balances con lógica de recarga automática
# ---------------------------------------------------------------------------
def corregir_balance(df_simulado: pd.DataFrame, df_estado: pd.DataFrame,
                     monto_recarga: float = 100_000) -> pd.DataFrame:
    df = df_simulado.merge(
        df_estado[["id_atm", "current_balance"]],
        on="id_atm", how="left"
    ).sort_values(["id_atm", "transaction_date"]).reset_index(drop=True)

    balances_finales  = []
    recargas_aplicadas = []

    for _, grupo in df.groupby("id_atm", sort=False):
        balance = float(grupo["current_balance"].iloc[0])

        for row in grupo.itertuples():
            monto  = float(row.simulated_amount) # type: ignore
            tipo   = row.transaction_type
            recarga = False

            if tipo == "WITHDRAWAL":
                balance = balance - monto if balance >= monto else 0.0
                if balance <= 0:
                    balance = monto_recarga
                    recarga = True

            elif tipo == "DEPOSIT":
                balance = min(balance + monto, monto_recarga)

            balances_finales.append(balance)
            recargas_aplicadas.append(recarga)

    df["balance_resultante"] = balances_finales
    df["recarga_aplicada"]   = recargas_aplicadas
    return df


# ---------------------------------------------------------------------------
# 4. Simular transacciones vía función SQL
# ---------------------------------------------------------------------------
def simular_transacciones(fecha_objetivo: str, df_estado: pd.DataFrame,
                          engine) -> pd.DataFrame:
    atm_ids      = df_estado["id_atm"].tolist()
    fecha_inicio = df_estado["last_transaction_date"].min() + pd.Timedelta(days=1)

    sql = """
    SELECT *
    FROM simulate_atm_transactions(%s, %s, %s, NULL)
    ORDER BY id_atm, transaction_date, transaction_type;
    """
    df_sim = pd.read_sql(sql, engine,params=(atm_ids, fecha_inicio, fecha_objetivo)) # type: ignore

    df_sim = buscar_impacto_climatico(df_sim)
    df_sim = corregir_balance(df_sim, df_estado)
    return df_sim


# ---------------------------------------------------------------------------
# 5. Insertar transacciones simuladas en daily_atm_transactions
# ---------------------------------------------------------------------------
def insert_transaction_data(df_simulado: pd.DataFrame, id_sync: int,
                            engine) -> int:
    if df_simulado.empty:
        log.warning("No hay datos para insertar.")
        return 0

    with engine.begin() as conn:
        max_id = conn.execute(
            text("SELECT COALESCE(MAX(id_transaction), 0) FROM public.daily_atm_transactions")
        ).scalar()

        df_ins = df_simulado.copy().reset_index(drop=True)
        df_ins["id_transaction"] = range(max_id + 1, max_id + 1 + len(df_ins))
        df_ins["id_sync"]        = id_sync
        df_ins["created_at"]     = datetime.now()
        df_ins = df_ins.rename(columns={"simulated_amount": "amount", "balance_resultante": "balance_after"})

        columnas = [
            "id_transaction", "id_atm", "transaction_date",
            "amount", "balance_after", "id_weather",
            "transaction_type", "id_sync", "created_at"
        ]
        df_ins[columnas].to_sql(
            name="daily_atm_transactions",
            con=conn,
            schema="public",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000
        )

    return len(df_ins)


# ---------------------------------------------------------------------------
# 6. Actualizar estado corriente de cada ATM
# ---------------------------------------------------------------------------
def update_atm_status(df_simulado: pd.DataFrame, id_sync: int,
                      engine) -> int:
    # Procesamiento inicial
    df_latest = df_simulado.sort_values(by=['id_atm', 'transaction_date']).groupby('id_atm').agg(
        current_balance=('balance_resultante', 'last'),
        last_transaction_date=('transaction_date', 'last'),
        last_deposit_date=('transaction_date', 
            lambda x: x[df_simulado.loc[x.index, 'transaction_type'] == 'DEPOSIT'].max()),
        last_reload_date=('transaction_date', 
            lambda x: x[df_simulado.loc[x.index, 'recarga_aplicada'] == True].max()),
        last_withdrawal_date=('transaction_date', 
            lambda x: x[df_simulado.loc[x.index, 'transaction_type'] == 'WITHDRAWAL'].max())
    ).reset_index()

    # --- CORRECCIÓN CLAVE AQUÍ ---
    # Convertimos las columnas de tiempo a tipo objeto para permitir el valor None puro
    for col in ['last_deposit_date', 'last_reload_date', 'last_withdrawal_date']:
        df_latest[col] = df_latest[col].astype(object)
        # Reemplazamos NaT por None explícitamente
        df_latest.loc[df_latest[col].isna(), col] = None
    # ------------------------------

    total = 0
    with engine.begin() as connection:
        for row in df_latest.itertuples(index=False):
            result = connection.execute(
                text("""
                    UPDATE public.atm_current_status
                        SET current_balance = :balance, 
                            last_sync_id = :sync_id, 
                            last_transaction_date = :t_date,
                            last_deposit_date = COALESCE(:d_date, last_deposit_date),
                            last_reload_date = COALESCE(:r_date, last_reload_date),
                            last_withdrawal_date = COALESCE(:w_date, last_withdrawal_date),
                            updated_at = NOW()
                        WHERE id_atm = :atm_id
                    """),
                    {
                        'balance': row.current_balance,
                        'sync_id': id_sync,
                        't_date': row.last_transaction_date,
                        'd_date': row.last_deposit_date, # Ahora sí enviará None (NULL en SQL)
                        'r_date': row.last_reload_date,
                        'w_date': row.last_withdrawal_date,
                        'atm_id': row.id_atm
                    }
            )
            total += result.rowcount

    return total


# ---------------------------------------------------------------------------
# 7. Gestión del registro en sync_logs (inicio / éxito / fallo)
# ---------------------------------------------------------------------------
def iniciar_sync(engine) -> int:
    with engine.begin() as conn:
        id_sync = conn.execute(
            text("""
                INSERT INTO public.sync_logs
                    (sync_start, status, source_system, process_log)
                VALUES (NOW(), 'IN_PROGRESS', 'CENTRAL DEL BANCO', CAST('[]' AS jsonb))
                RETURNING id_sync;
            """)
        ).scalar()
    return id_sync


def finalizar_sync(id_sync: int, total_inserted: int, total_updated: int,
                   audit: ProcessAudit, engine):
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE public.sync_logs
                SET records_inserted  = :inserted,
                    records_processed = :processed,
                    records_updated   = :updated,
                    sync_end          = NOW(),
                    status            = 'SUCCESS',
                    process_log       = CAST(:plog AS jsonb)
                WHERE id_sync = :sync_id;
            """),
            {
                "inserted":  total_inserted,
                "processed": total_inserted + total_updated,
                "updated":   total_updated,
                "plog":      audit.to_json_string(),
                "sync_id":   id_sync
            }
        )


def marcar_sync_fallido(id_sync: int, error: str, audit: ProcessAudit,
                        engine):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE public.sync_logs
                    SET sync_end      = NOW(),
                        status        = 'FAILED',
                        error_message = :error,
                        process_log   = CAST(:plog AS jsonb)
                    WHERE id_sync = :sync_id;
                """),
                {
                    "error":   error,
                    "plog":    audit.to_json_string(),
                    "sync_id": id_sync
                }
            )
    except Exception as ex:
        log.error("No se pudo actualizar sync_log como FAILED: %s", ex)

    
# Se debe agregar la actualizacion de la vista
def update_view_mv_historical_daily_atm(engine):
    with engine.begin() as conn:
        conn.execute(text("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_historical_daily_ATM;"))

def lanzar_monitoreo(engine)-> None:
    ejecutar_monitoreo(engine)

# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------
def ejecutar_sync(fecha_objetivo: str, engine):
    """
    Orquesta todo el proceso de sincronización.
    Devuelve un dict con el resultado listo para serializar a JSON.
    """

    # Inicialización del audit
    audit   = ProcessAudit()
    id_sync = None

    try:
        # — INICIO —
        id_sync = iniciar_sync(engine)
        audit.log("SYNC_START", id_sync=id_sync)

        # — ESTADO DE ATMs —
        df_estado = obtener_ultimo_estado_atm(engine)
        audit.log("FETCH_ATM_STATUS", atms_activos=len(df_estado))

        # — SIMULACIÓN —
        df_sim = simular_transacciones(fecha_objetivo, df_estado, engine)
        audit.log(
            "SIMULATE_TRANSACTIONS",
            filas_simuladas=len(df_sim),
            atms=df_sim["id_atm"].nunique(),
            recargas=int(df_sim["recarga_aplicada"].sum())
        )

        # — INSERCIÓN —
        total_inserted = insert_transaction_data(df_sim, id_sync, engine)
        audit.log("INSERT_TRANSACTIONS", inserted=total_inserted)

        # — ACTUALIZACIÓN STATUS —
        total_updated = update_atm_status(df_sim, id_sync, engine)
        audit.log("UPDATE_ATM_STATUS", atms_actualizados=total_updated)

        # — CIERRE EXITOSO —
        finalizar_sync(id_sync, total_inserted, total_updated, audit, engine)
        audit.log(
            "SYNC_END",
            total_records=total_inserted + total_updated
        )

        # — ACTUALIZACIÓN VISTA —
        update_view_mv_historical_daily_atm(engine)
        log.info("Vista actualizada exitosamente para la fecha: %s", fecha_objetivo)

        log.info("Generacion de datos completada exitosamente")
        log.info(f"Success: {True}, id_sync: {id_sync}, fecha_objetivo: {fecha_objetivo}, inserted: {total_inserted}, updated: {total_updated}, process_log: {audit.to_json_array()}")

        #-- Lanza Automaticamente el monitoreo
        lanzar_monitoreo(engine)
    except Exception as ex:
        log.error("Error en sincronización: %s", ex, exc_info=True)
        log.info(f"Success: {False}, id_sync: {id_sync}, fecha_objetivo: {fecha_objetivo}, error: {str(ex)}, process_log: {audit.to_json_array()}")
        audit.log("SYNC_ERROR", status="ERROR", error=str(ex))
        if id_sync:
            marcar_sync_fallido(id_sync, str(ex), audit, engine)