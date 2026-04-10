# src/fuga/data/data_extraction.py
"""
Extracción de datos de entrenamiento para el modelo de CHURN.
Fuente: account_details + customer + country + gender (PostgreSQL).
"""

import logging
import pandas as pd

from fuga.data.db_config import engine

logger = logging.getLogger(__name__)

CHURN_QUERY = """
SELECT
    ad.credit_score                                    AS "CreditScore",
    c.country_description                              AS "Geography",
    g.gender_description                               AS "Gender",
    cu.age                                             AS "Age",
    ad.tenure                                          AS "Tenure",
    ad.balance                                         AS "Balance",
    ad.num_of_products                                 AS "NumOfProducts",
    CASE WHEN ad.has_cr_card      = true THEN 1 ELSE 0 END AS "HasCrCard",
    CASE WHEN ad.is_active_member = true THEN 1 ELSE 0 END AS "IsActiveMember",
    ad.estimated_salary                                AS "EstimatedSalary",
    CASE WHEN ad.exited           = true THEN 1 ELSE 0 END AS "Exited"
FROM public.account_details ad
JOIN public.customer cu ON ad.id_customer  = cu.id_customer
JOIN public.country  c  ON cu.id_country   = c.id_country
JOIN public.gender   g  ON cu.id_gender    = g.id_gender
"""


def extract_training_data() -> pd.DataFrame:
    """
    Extrae todos los clientes de la BD para entrenamiento de churn usando SQLAlchemy engine.

    Returns:
        DataFrame con columnas: CreditScore, Geography, Gender, Age, Tenure,
        Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited.

    Raises:
        RuntimeError: Si no se puede conectar a la BD o hay menos de 100 registros.
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(CHURN_QUERY, conn)
        
        if df.empty:
            raise RuntimeError("La base de datos no devolvió registros de clientes.")

        # Limpieza Crítica: Asegurar que Exited sea numérico (evita errores de concatenación de strings)
        df['Exited'] = pd.to_numeric(df['Exited'], errors='coerce').fillna(0).astype(int)
        
        logger.info(f"Datos extraidos: {len(df)} registros, {df['Exited'].sum()} churn.")
        
        if len(df) < 100:
            raise RuntimeError(
                f"Datos insuficientes para entrenar: {len(df)} registros (minimo 100)."
            )
        return df
    except Exception:
        logger.exception("Error extrayendo datos de churn")
        raise
