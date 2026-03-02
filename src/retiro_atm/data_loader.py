# src/retiro_atm/data_loader.py
import logging
import pandas as pd
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


def consultar_ultima_version() -> int:
    """
    Consulta el ID máximo de la tabla de auditoría para calcular
    el siguiente version_tag.

    Returns:
        ID máximo o 0 si la tabla está vacía.
    """
    if database.engine is None:
        database.init_db()

    try:
        query = "SELECT COALESCE(MAX(id), 0) FROM self_training_audit_withdrawal_model"
        result = pd.read_sql(query, database.engine)
        return int(result.iloc[0, 0])
    except Exception as e:
        logger.warning(f"⚠️ No se pudo obtener última versión: {e}")
        return 0
