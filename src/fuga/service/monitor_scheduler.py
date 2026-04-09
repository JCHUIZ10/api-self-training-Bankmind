import os
import logging
import requests

logger = logging.getLogger(__name__)

_churn_scheduler = None

CHURN_MONITOR_ENABLED = os.environ.get("CHURN_MONITOR_ENABLED", "true").lower() == "true"
CHURN_MONITOR_INTERVAL_HOURS = int(os.environ.get("CHURN_MONITOR_INTERVAL_HOURS", "6"))
CHURN_EVALUATE_URL = os.environ.get("CHURN_EVALUATE_URL", "http://localhost:8001/fuga/monitor/evaluate")
CHURN_SELF_TRAINING_URL = os.environ.get("CHURN_SELF_TRAINING_URL", "http://localhost:8001/fuga/train")

def _run_churn_monitor_cycle():
    """
    Ciclo programado del monitor de rendimiento Churn.

    1. Llama a POST /fuga/monitor/evaluate en esta misma API (self-training).
       Ese endpoint evalúa recall vs ground truth y persiste el resultado.
    2. Si el status es 'degraded', dispara reentrenamiento en este servicio
       llamando a POST /fuga/train.
    """
    logger.info("[CHURN MONITOR] Iniciando ciclo programado → %s", CHURN_EVALUATE_URL)

    try:
        resp = requests.post(CHURN_EVALUATE_URL, timeout=60)
        if resp.status_code != 200:
            logger.warning(
                "[CHURN MONITOR] Evaluación retornó HTTP %s: %s",
                resp.status_code, resp.text[:200]
            )
            return

        status_data = resp.json()
        status = status_data.get("status", "unknown")
        recall = status_data.get("recall", "N/A")
        logger.info("[CHURN MONITOR] Estado del modelo: %s (recall=%s)", status, recall)

        if status == "degraded":
            logger.warning(
                "[CHURN MONITOR] Recall degradado detectado (%s). "
                "Disparando reentrenamiento automático...", recall
            )
            try:
                train_resp = requests.post(
                    CHURN_SELF_TRAINING_URL,
                    json={"triggered_by": "performance_monitor_decay"},
                    timeout=600,
                )
                if train_resp.status_code == 200:
                    result = train_resp.json()
                    logger.info(
                        "[CHURN MONITOR] Reentrenamiento completado — "
                        "promotion_status=%s", result.get("promotion_status")
                    )
                else:
                    logger.error(
                        "[CHURN MONITOR] Reentrenamiento retornó HTTP %s: %s",
                        train_resp.status_code, train_resp.text[:300]
                    )
            except requests.exceptions.ConnectionError:
                logger.error(
                    "[CHURN MONITOR] No se pudo conectar al self-training (%s). "
                    "¿Está corriendo?", CHURN_SELF_TRAINING_URL
                )
            except requests.exceptions.Timeout:
                logger.error("[CHURN MONITOR] Timeout esperando respuesta del self-training (>600s).")

        elif status == "healthy":
            logger.info("[CHURN MONITOR] Modelo saludable. No se requiere acción.")
        else:
            logger.info("[CHURN MONITOR] Estado: %s — se omite acción.", status)

    except requests.exceptions.ConnectionError:
        logger.warning(
            "[CHURN MONITOR] Endpoint de monitor no disponible en %s.", CHURN_EVALUATE_URL
        )
    except Exception as e:
        logger.error("[CHURN MONITOR] Error no controlado en ciclo: %s", e)

def setup_churn_monitor():
    """Configura el APScheduler para el monitor de rendimiento Churn."""
    global _churn_scheduler

    if not CHURN_MONITOR_ENABLED:
        logger.info("[CHURN MONITOR] Monitor deshabilitado (CHURN_MONITOR_ENABLED=false).")
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        _churn_scheduler = BackgroundScheduler()
        _churn_scheduler.add_job(
            _run_churn_monitor_cycle,
            'interval',
            hours=CHURN_MONITOR_INTERVAL_HOURS,
            id='churn_performance_monitor',
            name='Churn Performance Monitor',
            replace_existing=True,
            max_instances=1,
        )
        _churn_scheduler.start()
        logger.info(
            "[CHURN MONITOR] Scheduler iniciado — evaluación cada %sh.",
            CHURN_MONITOR_INTERVAL_HOURS
        )
    except ImportError:
        logger.warning(
            "[CHURN MONITOR] APScheduler no instalado. "
            "Instala con: pip install apscheduler"
        )
    except Exception as e:
        logger.error("[CHURN MONITOR] Error iniciando scheduler: %s", e)

def shutdown_churn_monitor():
    """Detiene el scheduler del monitor."""
    global _churn_scheduler
    if _churn_scheduler:
        _churn_scheduler.shutdown(wait=False)
        logger.info("[CHURN MONITOR] Scheduler detenido.")
        _churn_scheduler = None