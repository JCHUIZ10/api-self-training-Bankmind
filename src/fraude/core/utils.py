import logging
import math
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def get_reference_date() -> datetime:
    """
    Devuelve la fecha de referencia para el entrenamiento.

    Precedencia:
      1. Variable de entorno REFERENCE_DATE (YYYY-MM-DD)
      2. datetime.now()
    """
    raw = os.getenv("REFERENCE_DATE", "").strip()
    if raw:
        try:
            ref = datetime.strptime(raw, "%Y-%m-%d")
            logger.info("📅 REFERENCE_DATE desde entorno: %s", ref.date())
            return ref
        except ValueError:
            logger.warning(
                "⚠️ REFERENCE_DATE='%s' no es YYYY-MM-DD. Usando datetime.now().", raw
            )
    return datetime.now()


def compute_lambda(half_life_days: int) -> float:
    """
    Calcula λ a partir de la vida media.
    Fórmula: λ = ln(2) / half_life_days
    """
    lam = math.log(2) / half_life_days
    logger.info(
        "📐 Decay λ=%.6f (half_life=%d días → peso al año=%.1f%%)",
        lam, half_life_days, 100 * math.exp(-lam * 365),
    )
    return lam


def validate_training_dates(end_date: str, start_date: str | None = None) -> None:
    """
    Valida que las fechas de entrenamiento sean correctas.
    """
    try:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"end_date inválido. Use 'YYYY-MM-DD'. Error: {e}")

    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"start_date inválido. Use 'YYYY-MM-DD'. Error: {e}")

        if start_dt >= end_dt:
            raise ValueError("start_date debe ser anterior a end_date.")

        delta_days = (end_dt - start_dt).days
        if delta_days < 30:
            logger.warning(
                "⚠️ Rango de fechas muy corto (%d días). Recomendado: mínimo 90 días.", delta_days
            )
        logger.info("📅 Rango validado: %d días (%s → %s)", delta_days, start_date, end_date)
    else:
        logger.info("📅 end_date validado: %s (start_date se calculará automáticamente)", end_date)
