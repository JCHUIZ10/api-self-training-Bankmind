# src/fraude/data_extraction.py
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from fraude.db_config import get_db_connection

logger = logging.getLogger(__name__)


def extract_and_balance_data(
    start_date: str,
    end_date: str,
    undersampling_ratio: int = 4
) -> pd.DataFrame:
    """
    Extrae datos de operational_transactions con sampling balanceado.
    
    Estrategia:
    1. Extrae el 100% de fraudes (clase minoritaria)
    2. Extrae una muestra aleatoria de transacciones legítimas
    3. Ratio configurable: por defecto 4 legítimas por cada fraude
    
    Args:
        start_date: Fecha inicial (formato: 'YYYY-MM-DD')
        end_date: Fecha final (formato: 'YYYY-MM-DD')
        undersampling_ratio: Número de legítimas por cada fraude (default: 4)
    
    Returns:
        DataFrame con datos balanceados
    """
    conn = get_db_connection()
    
    try:
        logger.info(f"📊 Extrayendo datos del {start_date} al {end_date}")
        
        # =========================================
        # 1. EXTRAER TODOS LOS FRAUDES
        # =========================================
        fraud_query = """
            SELECT 
                ot.amt,
                l.city_pop,
                c.category_name as category,
                g.gender_description as gender,
                cu.job,
                l.customer_lat as lat,
                l.customer_long as long,
                ot.merch_lat,
                ot.merch_long,
                ot.trans_date_time::text as trans_date_trans_time,
                cu.dob::text,
                ot.is_fraud_ground_truth as is_fraud,
                1.0 as sample_weight
            FROM operational_transactions ot
            JOIN credit_cards cc ON ot.cc_num = cc.cc_num
            JOIN customer cu ON cc.id_customer = cu.id_customer
            JOIN localization l ON cu.id_localization = l.id_localization
            JOIN gender g ON cu.id_gender = g.id_gender
            JOIN categories c ON ot.id_category = c.id_category
            WHERE ot.is_fraud_ground_truth = 1
              AND ot.trans_date_time BETWEEN %s AND %s
        """
        
        # Ejecutar fraud query usando cursor directamente
        cursor = conn.cursor()
        cursor.execute(fraud_query, [start_date, end_date])
        columns = [desc[0] for desc in cursor.description]
        fraud_rows = cursor.fetchall()
        cursor.close()
        
        # Construir DataFrame manualmente
        df_fraud = pd.DataFrame(fraud_rows, columns=columns)
        fraud_count = len(df_fraud)
        
        logger.info(f"🚨 Fraudes encontrados: {fraud_count}")
        
        if fraud_count == 0:
            raise ValueError("No se encontraron fraudes en el rango de fechas especificado")
        
        # =========================================
        # 2. CONTAR TOTAL DE LEGÍTIMAS
        # =========================================
        count_query = """
            SELECT COUNT(*) as total
            FROM operational_transactions ot
            JOIN credit_cards cc ON ot.cc_num = cc.cc_num
            JOIN customer cu ON cc.id_customer = cu.id_customer
            JOIN localization l ON cu.id_localization = l.id_localization
            JOIN gender g ON cu.id_gender = g.id_gender
            JOIN categories c ON ot.id_category = c.id_category
            WHERE ot.is_fraud_ground_truth = 0
              AND ot.trans_date_time BETWEEN %s AND %s
        """
        
        
        # Ejecutar COUNT query usando cursor directamente
        cursor = conn.cursor()
        cursor.execute(count_query, [start_date, end_date])
        result_row = cursor.fetchone()
        cursor.close()
        

        
        # Extraer el valor dependiendo del tipo de resultado
        if isinstance(result_row, dict):
            total_legitimate = int(result_row['total'])
        elif isinstance(result_row, (tuple, list)):
            total_legitimate = int(result_row[0])
        else:
            raise ValueError(f"Tipo inesperado de result_row: {type(result_row)}, contenido: {result_row}")
        
        logger.info(f"✅ Legítimas totales disponibles: {total_legitimate:,}")
        
        # =========================================
        # 3. CALCULAR TAMAÑO DE MUESTRA
        # =========================================
        legitimate_sample_size = fraud_count * undersampling_ratio
        
        # Verificar que no se pida más de lo disponible
        if legitimate_sample_size > total_legitimate:
            logger.warning(f"⚠️ Muestra solicitada ({legitimate_sample_size}) > disponible ({total_legitimate}). "
                          f"Usando todas las legítimas disponibles.")
            legitimate_sample_size = total_legitimate
        
        logger.info(f"📐 Tomando {legitimate_sample_size:,} legítimas "
                   f"(ratio {undersampling_ratio}:1)")
        
        # =========================================
        # 4. SAMPLING ALEATORIO DE LEGÍTIMAS
        # =========================================
        legitimate_query = """
            SELECT 
                ot.amt,
                l.city_pop,
                c.category_name as category,
                g.gender_description as gender,
                cu.job,
                l.customer_lat as lat,
                l.customer_long as long,
                ot.merch_lat,
                ot.merch_long,
                ot.trans_date_time::text as trans_date_trans_time,
                cu.dob::text,
                ot.is_fraud_ground_truth as is_fraud,
                1.0 as sample_weight
            FROM operational_transactions ot
            JOIN credit_cards cc ON ot.cc_num = cc.cc_num
            JOIN customer cu ON cc.id_customer = cu.id_customer
            JOIN localization l ON cu.id_localization = l.id_localization
            JOIN gender g ON cu.id_gender = g.id_gender
            JOIN categories c ON ot.id_category = c.id_category
            WHERE ot.is_fraud_ground_truth = 0
              AND ot.trans_date_time BETWEEN %s AND %s
            ORDER BY RANDOM()
            LIMIT %s
        """
        
        # Ejecutar legitimate query usando cursor directamente
        cursor = conn.cursor()
        cursor.execute(legitimate_query, [start_date, end_date, legitimate_sample_size])
        columns = [desc[0] for desc in cursor.description]
        legitimate_rows = cursor.fetchall()
        cursor.close()
        
        # Construir DataFrame manualmente
        df_legitimate = pd.DataFrame(legitimate_rows, columns=columns)
        
        logger.info(f"✅ Legítimas extraídas: {len(df_legitimate):,}")
        
        # =========================================
        # 5. COMBINAR Y MEZCLAR
        # =========================================
        df_combined = pd.concat([df_fraud, df_legitimate], ignore_index=True)
        
        # IMPORTANTE: Convertir tipos de datos
        # Cuando se crea DataFrame desde cursor, todas las columnas son 'object' (string)
        # Necesitamos convertir explícitamente las numéricas
        numeric_cols = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 
                       'is_fraud', 'sample_weight']
        
        for col in numeric_cols:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        
        logger.info(f"✅ Tipos de datos convertidos correctamente")
        
        # Mezclar aleatoriamente
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calcular estadísticas finales
        final_fraud_ratio = len(df_fraud) / len(df_combined)
        
        logger.info(f"📊 Dataset balanceado creado:")
        logger.info(f"   - Total samples: {len(df_combined):,}")
        logger.info(f"   - Fraudes: {len(df_fraud):,} ({final_fraud_ratio:.1%})")
        logger.info(f"   - Legítimas: {len(df_legitimate):,} ({1-final_fraud_ratio:.1%})")
        logger.info(f"   - Ratio final: {len(df_legitimate)/len(df_fraud):.2f}:1")
        
        return df_combined
        
    except Exception as e:
        logger.error(f"❌ Error extrayendo datos: {e}")
        raise
    finally:
        conn.close()


def validate_date_range(start_date: str, end_date: str):
    """
    Valida que el rango de fechas sea correcto.
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start >= end:
            raise ValueError("start_date debe ser anterior a end_date")
        
        # Validar que el rango no sea muy pequeño (al menos 1 mes)
        delta_days = (end - start).days
        if delta_days < 30:
            logger.warning(f"⚠️ Rango de fechas muy corto: {delta_days} días. "
                          f"Recomendado: al menos 90 días para tener suficientes fraudes.")
        
        # Validar que el rango no sea muy grande (máximo 6 meses)
        if delta_days > 180:
            logger.warning(f"⚠️ Rango de fechas muy amplio: {delta_days} días. "
                          f"Esto puede generar un dataset muy grande.")
        
        logger.info(f"📅 Rango de fechas validado: {delta_days} días")
        
    except ValueError as e:
        raise ValueError(f"Formato de fecha inválido. Use 'YYYY-MM-DD'. Error: {e}")


def get_raw_transactions(start_date: str, end_date: str) -> 'pd.DataFrame':
    """
    Extrae transacciones sin balancear (todas), para cálculos de drift / PSI.
    A diferencia de extract_and_balance_data, NO aplica undersampling.

    Args:
        start_date: 'YYYY-MM-DD'
        end_date:   'YYYY-MM-DD'

    Returns:
        DataFrame con todas las transacciones del período (fraudes + legítimas).
    """
    conn = get_db_connection()
    try:
        query = """
            SELECT
                ot.amt,
                l.city_pop,
                c.category_name  AS category,
                g.gender_description AS gender,
                cu.job,
                l.customer_lat   AS lat,
                l.customer_long  AS long,
                ot.merch_lat,
                ot.merch_long,
                ot.trans_date_time::text AS trans_date_trans_time,
                cu.dob::text,
                ot.is_fraud_ground_truth AS is_fraud
            FROM operational_transactions ot
            JOIN credit_cards cc  ON ot.cc_num      = cc.cc_num
            JOIN customer cu      ON cc.id_customer  = cu.id_customer
            JOIN localization l   ON cu.id_localization = l.id_localization
            JOIN gender g         ON cu.id_gender    = g.id_gender
            JOIN categories c     ON ot.id_category  = c.id_category
            WHERE ot.trans_date_time BETWEEN %s AND %s
        """
        cursor = conn.cursor()
        cursor.execute(query, [start_date, end_date])
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(rows, columns=columns)

        # Convertir tipos numéricos
        for col in ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'is_fraud']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        logger.info(f"📊 get_raw_transactions: {len(df):,} filas ({start_date} → {end_date})")
        return df

    except Exception as e:
        logger.error(f"❌ Error en get_raw_transactions: {e}")
        raise
    finally:
        conn.close()

