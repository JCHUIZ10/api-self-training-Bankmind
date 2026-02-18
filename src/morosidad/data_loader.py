import pandas as pd
import logging
from morosidad import database

logger = logging.getLogger(__name__)

def load_training_data() -> pd.DataFrame:
    """
    Carga los datos de entrenamiento desde la vista materializada en PostgreSQL.
    Retorna un DataFrame de Pandas listo para el entrenamiento.
    """
    if database.engine is None:
        database.init_db()
        
    query = """
    SELECT 
        limit_bal, sex, education, marriage, age, 
        pay_0, pay_2, pay_3, pay_4, pay_5, pay_6, 
        bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6, 
        pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6, 
        utilization_rate, default_payment_next_month, sample_weight
    FROM vw_training_dataset_morosidad
    """
    
    logger.info("📊 Iniciando extracción de datos desde PostgreSQL...")
    try:
        df = pd.read_sql(query, database.engine)
        df.columns = df.columns.str.upper()
        logger.info(f"✅ Datos cargados exitosamente. Total registros: {len(df)}")
        
        if df.empty:
            logger.warning("⚠️ La vista materializada retornó 0 filas.")
            return None
            
        return df
    except Exception as e:
        logger.error(f"❌ Error al cargar datos de la BD: {e}")
        raise e


def get_dataset_start_date() -> str:
    """
    Obtiene la fecha más antigua (feature_period) del dataset de entrenamiento.
    Retorna un string ISO format o None si no se puede obtener.
    """
    if database.engine is None:
        database.init_db()

    query = """
    SELECT MIN(monthly_period) as min_date
    FROM monthly_history
    WHERE monthly_period >= CURRENT_DATE - INTERVAL '3 years'
    """
    try:
        result = pd.read_sql(query, database.engine)
        min_date = result['min_date'].iloc[0]
        if min_date is not None:
            return str(min_date)
        return None
    except Exception as e:
        logger.warning(f"⚠️ No se pudo obtener fecha inicio del dataset: {e}")
        return None
