# src/fraude/data_extraction.py
import logging
import math
import os
from datetime import datetime, timedelta

import pandas as pd

from fraude.data.db_config import get_db_connection

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes por defecto
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_MAX_HISTORY_DAYS = 730   # 2 años
DEFAULT_HALF_LIFE_DAYS   = 180   # peso 0.5 a los 180 días
DEFAULT_IF_RECENT_MONTHS = 6     # ventana de IsolationForest

# ─────────────────────────────────────────────────────────────────────────────
# DataProvider — Separación de Responsabilidades (SRP)
# ─────────────────────────────────────────────────────────────────────────────

class DataProvider:
    """
    Provee vistas del DataFrame de entrenamiento a los distintos modelos del
    pipeline híbrido.

    Responsabilidad única: saber qué porción del dataset debe ver cada modelo.

    - XGBoost        → toda la historia con pesos exponenciales (get_full_data).
    - IsolationForest→ solo los últimos N meses, sin pesos (get_recent_data).
      Justificación: IF no soporta sample_weight; limitarlo al período reciente
      garantiza que el anomaly_score refleje los patrones de fraude actuales.

    Nota: el feature engineering se realiza FUERA de esta clase, en
    training_service.py, para mantener la separación entre extracción y
    transformación.
    """

    def __init__(self, df: pd.DataFrame, if_recent_months: int = DEFAULT_IF_RECENT_MONTHS) -> None:
        if "trans_date_trans_time" not in df.columns:
            raise ValueError(
                "DataProvider requiere la columna 'trans_date_trans_time' (datetime) en el DataFrame."
            )
        self._df = df
        self._if_recent_months = if_recent_months

    def get_full_data(self) -> pd.DataFrame:
        """
        Devuelve el dataset completo (toda la historia) con pesos exponenciales.
        Usado por XGBoost para entrenamiento ponderado.
        """
        return self._df

    def get_recent_data(self) -> pd.DataFrame:
        """
        Devuelve únicamente los datos de los últimos `if_recent_months` meses.
        Usado por IsolationForest, que no soporta sample_weight.
        """
        max_date = self._df["trans_date_trans_time"].max()
        cutoff   = max_date - pd.DateOffset(months=self._if_recent_months)
        recent   = self._df[self._df["trans_date_trans_time"] >= cutoff].copy()
        logger.info(
            "🌳 DataProvider.get_recent_data(): %d registros (últimos %d meses, corte=%s)",
            len(recent), self._if_recent_months, cutoff.date(),
        )
        return recent


# ─────────────────────────────────────────────────────────────────────────────
# Extracción principal
# ─────────────────────────────────────────────────────────────────────────────

def extract_training_data(
    end_date: str,
    lam: float,
    max_history_days: int = DEFAULT_MAX_HISTORY_DAYS,
    undersampling_ratio: int = 4,
    start_date: str | None = None,
) -> pd.DataFrame:
    """
    Extrae datos de `operational_transactions` con decay temporal exponencial
    calculado directamente en PostgreSQL.

    Estrategia de extracción:
    ─────────────────────────
    1. El peso de cada transacción se calcula en la query como:
           weight = EXP(-λ * days_since_transaction)
       siendo `end_date` el punto de referencia (peso 1.0 en ese día).
    2. Fraudes: se extrae el 100% del período (clase minoritaria).
    3. Legítimas: undersampling count-based (fraud_count × undersampling_ratio).
       El propio XGBoost aplicará los pesos temporales durante el entrenamiento,
       por lo que el ratio count-based es suficiente.

    Args:
        end_date:           Fecha de referencia ('YYYY-MM-DD'). Peso 1.0 en este día.
        lam:                Lambda del decay (calculado de half_life_days).
        max_history_days:   Techo de historia en días (solo si start_date is None).
        undersampling_ratio:Número de legítimas por cada fraude.
        start_date:         Límite inferior opcional. Si None, se calcula
                            como end_date - max_history_days.

    Returns:
        DataFrame balanceado con columna `sample_weight` ya calculada.
    """
    conn = get_db_connection()
    try:
        # ── Calcular límite inferior ──────────────────────────────────────────
        end_dt   = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = (
            datetime.strptime(start_date, "%Y-%m-%d")
            if start_date
            else end_dt - timedelta(days=max_history_days)
        )
        effective_start = start_dt.strftime("%Y-%m-%d")
        logger.info(
            "📊 Extrayendo datos del %s al %s (λ=%.6f)",
            effective_start, end_date, lam,
        )

        # ── 1. Fraudes (100 %) ────────────────────────────────────────────────
        # El decay se calcula en SQL para delegar la operación pesada a PostgreSQL.
        # Fórmula: EXP(-λ * días_desde_transacción_hasta_end_date)
        # El multiplicador de intervalo `%s * INTERVAL '1 day'` es SQL válido en PG.
        fraud_query = """
            SELECT
                ot.amt,
                l.city_pop,
                c.category_name            AS category,
                g.gender_description       AS gender,
                cu.job,
                l.customer_lat             AS lat,
                l.customer_long            AS long,
                ot.merch_lat,
                ot.merch_long,
                ot.trans_date_time::text   AS trans_date_trans_time,
                cu.dob::text,
                ot.is_fraud_ground_truth   AS is_fraud,
                EXP(
                    -(%s::float)
                    * DATE_PART('day', %s::timestamp - ot.trans_date_time)
                )                          AS sample_weight
            FROM operational_transactions ot
            JOIN credit_cards cc ON ot.cc_num          = cc.cc_num
            JOIN customer     cu ON cc.id_customer     = cu.id_customer
            JOIN localization  l ON cu.id_localization = l.id_localization
            JOIN gender        g ON cu.id_gender       = g.id_gender
            JOIN categories    c ON ot.id_category     = c.id_category
            WHERE ot.is_fraud_ground_truth = 1
              AND ot.trans_date_time BETWEEN %s AND %s
        """

        cursor = conn.cursor()
        cursor.execute(fraud_query, [lam, end_date, effective_start, end_date])
        columns    = [desc[0] for desc in cursor.description]
        fraud_rows = cursor.fetchall()
        cursor.close()

        df_fraud    = pd.DataFrame(fraud_rows, columns=columns)
        fraud_count = len(df_fraud)
        logger.info("🚨 Fraudes encontrados: %d", fraud_count)

        if fraud_count == 0:
            raise ValueError(
                f"No se encontraron fraudes entre {effective_start} y {end_date}. "
                "Amplía el rango o verifica la tabla operational_transactions."
            )

        # ── 2. Contar legítimas disponibles ───────────────────────────────────
        count_query = """
            SELECT COUNT(*) AS total
            FROM operational_transactions ot
            JOIN credit_cards cc ON ot.cc_num          = cc.cc_num
            JOIN customer     cu ON cc.id_customer     = cu.id_customer
            JOIN localization  l ON cu.id_localization = l.id_localization
            JOIN gender        g ON cu.id_gender       = g.id_gender
            JOIN categories    c ON ot.id_category     = c.id_category
            WHERE ot.is_fraud_ground_truth = 0
              AND ot.trans_date_time BETWEEN %s AND %s
        """
        cursor = conn.cursor()
        cursor.execute(count_query, [effective_start, end_date])
        result_row      = cursor.fetchone()
        cursor.close()

        if isinstance(result_row, dict):
            total_legitimate = int(result_row["total"])
        elif isinstance(result_row, (tuple, list)):
            total_legitimate = int(result_row[0])
        else:
            raise ValueError(
                f"Tipo inesperado de result_row: {type(result_row)}, contenido: {result_row}"
            )
        logger.info("✅ Legítimas disponibles en el período: %d", total_legitimate)

        # ── 3. Calcular tamaño de muestra ─────────────────────────────────────
        legitimate_sample_size = fraud_count * undersampling_ratio
        if legitimate_sample_size > total_legitimate:
            logger.warning(
                "⚠️ Muestra solicitada (%d) > disponible (%d). Usando todas las legítimas.",
                legitimate_sample_size, total_legitimate,
            )
            legitimate_sample_size = total_legitimate

        logger.info(
            "📐 Extrayendo %d legítimas (ratio %d:1 count-based; XGBoost aplica pesos temporales)",
            legitimate_sample_size, undersampling_ratio,
        )

        # ── 4. Legítimas con sampling aleatorio y decay en SQL ────────────────
        legitimate_query = """
            SELECT
                ot.amt,
                l.city_pop,
                c.category_name            AS category,
                g.gender_description       AS gender,
                cu.job,
                l.customer_lat             AS lat,
                l.customer_long            AS long,
                ot.merch_lat,
                ot.merch_long,
                ot.trans_date_time::text   AS trans_date_trans_time,
                cu.dob::text,
                ot.is_fraud_ground_truth   AS is_fraud,
                EXP(
                    -(%s::float)
                    * DATE_PART('day', %s::timestamp - ot.trans_date_time)
                )                          AS sample_weight
            FROM operational_transactions ot
            JOIN credit_cards cc ON ot.cc_num          = cc.cc_num
            JOIN customer     cu ON cc.id_customer     = cu.id_customer
            JOIN localization  l ON cu.id_localization = l.id_localization
            JOIN gender        g ON cu.id_gender       = g.id_gender
            JOIN categories    c ON ot.id_category     = c.id_category
            WHERE ot.is_fraud_ground_truth = 0
              AND ot.trans_date_time BETWEEN %s AND %s
            ORDER BY RANDOM()
            LIMIT %s
        """
        cursor = conn.cursor()
        cursor.execute(
            legitimate_query,
            [lam, end_date, effective_start, end_date, legitimate_sample_size],
        )
        columns         = [desc[0] for desc in cursor.description]
        legitimate_rows = cursor.fetchall()
        cursor.close()

        df_legitimate = pd.DataFrame(legitimate_rows, columns=columns)
        logger.info("✅ Legítimas extraídas: %d", len(df_legitimate))

        # ── 5. Combinar, convertir tipos y mezclar ────────────────────────────
        df_combined = pd.concat([df_fraud, df_legitimate], ignore_index=True)

        numeric_cols = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long",
                        "is_fraud", "sample_weight"]
        for col in numeric_cols:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")

        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

        # ── 6. Log de pesos para verificación ────────────────────────────────
        weights      = df_combined["sample_weight"]
        fraud_ratio  = len(df_fraud) / len(df_combined)
        logger.info("📊 Dataset balanceado con decay temporal:")
        logger.info("   - Total samples    : %d", len(df_combined))
        logger.info("   - Fraudes          : %d (%.1f%%)", len(df_fraud), fraud_ratio * 100)
        logger.info("   - Legítimas        : %d (%.1f%%)", len(df_legitimate), (1 - fraud_ratio) * 100)
        logger.info("   - sample_weight min: %.4f", weights.min())
        logger.info("   - sample_weight mean: %.4f", weights.mean())
        logger.info("   - sample_weight max: %.4f", weights.max())
        logger.info("   - Rango efectivo   : %s → %s", effective_start, end_date)

        return df_combined

    except Exception:
        logger.exception("❌ Error extrayendo datos de entrenamiento")
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Extracción sin balancear (PSI / drift)
# ─────────────────────────────────────────────────────────────────────────────

def get_raw_transactions(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Extrae transacciones sin balancear (todas), para cálculos de drift / PSI.
    A diferencia de extract_training_data, NO aplica undersampling ni decay.

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
            JOIN credit_cards cc  ON ot.cc_num        = cc.cc_num
            JOIN customer     cu  ON cc.id_customer   = cu.id_customer
            JOIN localization  l  ON cu.id_localization = l.id_localization
            JOIN gender        g  ON cu.id_gender     = g.id_gender
            JOIN categories    c  ON ot.id_category   = c.id_category
            WHERE ot.trans_date_time BETWEEN %s AND %s
        """
        cursor = conn.cursor()
        cursor.execute(query, [start_date, end_date])
        columns = [desc[0] for desc in cursor.description]
        rows    = cursor.fetchall()
        cursor.close()

        df = pd.DataFrame(rows, columns=columns)

        for col in ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "is_fraud"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("📊 get_raw_transactions: %d filas (%s → %s)", len(df), start_date, end_date)
        return df

    except Exception:
        logger.exception("❌ Error en get_raw_transactions")
        raise
    finally:
        conn.close()