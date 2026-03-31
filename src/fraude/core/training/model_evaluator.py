import time
import logging
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from fraude.schemas.fraude import TrainingMetrics

logger = logging.getLogger(__name__)

def evaluate_model(xgb_model, X_test, y_test, w_test, start_time: float) -> tuple[TrainingMetrics, float]:
    logger.info("🎯 Calculando threshold óptimo (Recall >= 95%)...")
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    target_recall  = 0.95
    valid_indices  = np.where(recalls >= target_recall)[0]

    if len(valid_indices) > 0:
        best_idx       = valid_indices[np.argmax(precisions[valid_indices])]
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.5

    y_pred_optimizado = (y_prob >= best_threshold).astype(int)

    logger.info("📏 Calculando métricas finales...")
    auc_roc       = roc_auc_score(y_test, y_prob, sample_weight=w_test)
    training_time = time.time() - start_time

    metrics = TrainingMetrics(
        auc_roc           = round(float(auc_roc), 4),
        accuracy          = round(float(accuracy_score(y_test, y_pred_optimizado, sample_weight=w_test)), 4),
        precision         = round(float(precision_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0)), 4),
        recall            = round(float(recall_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0)), 4),
        f1_score          = round(float(f1_score(y_test, y_pred_optimizado, sample_weight=w_test, zero_division=0)), 4),
        optimal_threshold = round(best_threshold, 4),
        training_time_sec = round(training_time, 2),
    )
    
    logger.info("\n--- Reporte de Clasificación ---\n%s", classification_report(y_test, y_pred_optimizado))
    
    return metrics, best_threshold

def create_shap_explainer(xgb_model):
    logger.info("🔬 Creando SHAP explainer...")
    try:
        explainer = shap.TreeExplainer(xgb_model)
        logger.info("✅ SHAP explainer creado correctamente")
        return explainer
    except Exception:
        logger.warning("⚠️ No se pudo crear SHAP explainer", exc_info=True)
        return None

def compute_baseline_distributions(df: pd.DataFrame, numeric_features: list) -> dict:
    logger.info("📐 Calculando baseline_distributions para PSI drift...")
    baseline_distributions = {}
    PERCENTILES = list(range(10, 100, 10))

    for feat in numeric_features:
        try:
            if feat not in df.columns:
                continue
            values = df[feat].dropna().values
            if len(values) == 0:
                continue
            pct_edges = np.percentile(values, PERCENTILES).tolist()
            bins      = [float(values.min())] + pct_edges + [float(values.max())]
            counts, _ = np.histogram(values, bins=np.unique(bins))
            total     = counts.sum()
            pct       = (counts / total * 100.0).tolist() if total > 0 else [0.0] * len(counts)
            baseline_distributions[feat] = {
                "bins":      [round(b, 6) for b in np.unique(bins).tolist()],
                "pct":       [round(p, 6) for p in pct],
                "n_samples": int(total),
            }
        except Exception:
            logger.warning("   ⚠️ No se pudo calcular baseline para '%s'", feat, exc_info=True)
            
    return baseline_distributions