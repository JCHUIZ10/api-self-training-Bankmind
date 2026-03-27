import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def evaluate_promotion(challenger_metrics: dict, champion_metrics: Optional[dict]) -> Tuple[str, str]:
    """
    Compara las métricas del modelo retador contra el campeón actual (si existe).
    Retorna (promotion_status, promotion_reason).
    Lógica de negocio pura independiente de base de datos.
    """
    if not champion_metrics:
        logger.info("🎉 Primer modelo del sistema o champion sin métricas, promoción automática")
        return "PROMOTED", "Champion sin métricas registradas, promoción automática"

    f1_diff     = challenger_metrics["f1_score"] - champion_metrics["f1_score"]
    recall_diff = challenger_metrics["recall"]   - champion_metrics["recall"]

    logger.info("   Champion F1=%.4f | Challenger F1=%.4f", champion_metrics["f1_score"], challenger_metrics["f1_score"])
    logger.info("   Champion Recall=%.4f | Challenger Recall=%.4f", champion_metrics["recall"], challenger_metrics["recall"])

    if recall_diff >= 0 and f1_diff >= 0:
        logger.info("✅ PROMOTED: Challenger supera al champion")
        return "PROMOTED", f"Mejor rendimiento: F1 +{f1_diff:.4f}, Recall +{recall_diff:.4f}"
    
    elif recall_diff >= -0.01 and f1_diff > 0.005:
        logger.info("✅ PROMOTED: F1 mejora compensa ligera caída de recall")
        return "PROMOTED", f"F1 mejorado (+{f1_diff:.4f}), Recall aceptable ({recall_diff:+.4f})"
    
    else:
        logger.info("❌ REJECTED: Challenger no supera al champion")
        return "REJECTED", f"Rendimiento insuficiente: F1 {f1_diff:+.4f}, Recall {recall_diff:+.4f}"
