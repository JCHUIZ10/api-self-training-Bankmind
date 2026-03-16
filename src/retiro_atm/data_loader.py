# src/retiro_atm/data_loader.py
import logging
import pandas as pd
from sqlalchemy import text
from retiro_atm import database

logger = logging.getLogger(__name__)

DATASET_VIEW = "v_dataset_atm"
NAN_CHECK_COLUMNS = 11  # Primeras 11 columnas deben estar completas


def load_dataset() -> pd.DataFrame:
    """
    Carga el dataset de retiros ATM desde la vista materializada,
    aplica limpieza básica y retorna un DataFrame listo para preprocesamiento.

    Returns:
        DataFrame con datos limpios y fecha_transaccion como datetime.

    Raises:
        ValueError: Si no se obtienen datos.
    """
    if database.engine is None:
        database.init_db()

    logger.info(f"📊 Cargando dataset desde '{DATASET_VIEW}'...")

    try:
        query = f"SELECT * FROM {DATASET_VIEW}"
        df = pd.read_sql(query, database.engine)

        if df.empty:
            raise ValueError(f"La vista '{DATASET_VIEW}' retornó 0 filas.")

        logger.info(f"✅ Datos cargados: {df.shape[0]} filas x {df.shape[1]} columnas")

        # Limpiar NaN en las primeras N columnas (features obligatorias)
        filas_antes = len(df)
        df = df.dropna(subset=df.columns[:NAN_CHECK_COLUMNS])
        filas_eliminadas = filas_antes - len(df)
        if filas_eliminadas > 0:
            logger.info(f"🧹 Eliminadas {filas_eliminadas} filas con NaN en columnas clave")

        # Asegurar tipo datetime
        df["fecha_transaccion"] = pd.to_datetime(df["fecha_transaccion"])

        logger.info(
            f"📅 Rango: {df['fecha_transaccion'].min().date()} "
            f"→ {df['fecha_transaccion'].max().date()}"
        )
        return df

    except Exception as e:
        logger.error(f"❌ Error al cargar datos: {e}")
        raise


def consultar_ultima_version_modelo(nombre_modelo:str) -> int:
    """
    Retorna la ultima version del modelo contabilizando si existen versiones
    del modelo en la tabla de auditoria.
    
    Returns:
        int: ultima version del modelo
    """
    if database.engine is None:
        database.init_db()

    try:
        # 1. Usamos text() para declarar la consulta
        query = text(""" 
            SELECT COUNT(stawm.model_name) 
            FROM self_training_audit_withdrawal_model stawm 
            WHERE stawm.model_name ILIKE '%' || :model_name || '%'
        """)
        
        # 2. Ejecutamos pasando el diccionario de parámetros
        # Usamos una conexión explícita (buena práctica)
        with database.engine.connect() as conn:
            result = pd.read_sql(query, conn, params={"model_name": nombre_modelo})
            
        return int(result.iloc[0, 0])
    except Exception as e:
        logger.warning(f"⚠️ No se pudo obtener última versión: {e}")
        return 0

def obtener_distribucion_actual_atm_features(rango_muestra: int =60):
    """
    Este metodo obtiene una muestra actual de la tama atm_feactures 
    de los datos no estacionarios desde la ultima fecha registrada hasta
    el numero de dias a tras definido por el rango_muestra

    rango_muestra : numero de dias para seleccion de muestra desde el ultimo registro atm_feacture
    """
    if database.engine is None:
        database.init_db()

    try:
        query_psi_data = text("""
                WITH max_fecha AS (
                    SELECT MAX(reference_date) AS max_date
                    FROM atm_features
                )
                SELECT
                    (af.dynamic_features->>'lag1')::float AS lag1,
                    (af.dynamic_features->>'lag5')::float AS lag5,
                    (af.dynamic_features->>'lag11')::float AS lag11,
                    (af.dynamic_features->>'tendencia_lags')::float AS tendencia_lags,
                    (af.dynamic_features->>'ratio_finde_vs_semana')::float AS ratio_finde_vs_semana,
                    (af.dynamic_features->>'retiros_finde_anterior')::float AS retiros_finde_anterior,
                    (af.dynamic_features->>'retiros_domingo_anterior')::float AS retiros_domingo_anterior
                FROM atm_features af
                CROSS JOIN max_fecha mf
                WHERE af.reference_date >= mf.max_date - INTERVAL '1 day' * :dias
            """)

        # Using a context manager for the connection is safer for resource management
        with database.engine.connect() as conn:
            return pd.read_sql(query_psi_data, conn, params={"dias": rango_muestra})
            
    except Exception as e: 
        logger.error(f"No se pudo cargar la distribución actual para el PSI: {e}")
        raise Exception(e)