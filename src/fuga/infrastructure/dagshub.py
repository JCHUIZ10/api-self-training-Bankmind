# src/fuga/infrastructure/dagshub.py
"""
Integración con DagsHub para el módulo CHURN.
Sube el combo-pack (modelo + scaler + features) y verifica integridad.
"""

import io
import logging
import os

import dagshub
import joblib
import mlflow
import requests

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN HARDCODEADA — DagsHub Churn (Self-Training)
# ============================================================
DAGSHUB_REPO_OWNER = "notificacionesbankmind"
DAGSHUB_REPO_NAME = "Modelos_BankMind_2026"
DAGSHUB_MODEL_PATH = "modelos/fuga/modelo.pkl"
DAGSHUB_TOKEN = "1022993058d503226b5e83a649a067c0c2ef2e73"

os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN

_dagshub_initialized = False


def init_dagshub_connection():
    global _dagshub_initialized
    if _dagshub_initialized:
        return
    if not DAGSHUB_TOKEN:
        logger.warning("DAGSHUB_USER_TOKEN no configurado — MLflow no funcionara remotamente.")
        return
    try:
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )
        _dagshub_initialized = True
        logger.info(f"Conectado a DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
    except Exception as e:
        logger.error(f"Error inicializando DagsHub: {e}")


def upload_champion(model_bytes: bytes, version_tag: str):
    """
    Sube el nuevo champion de churn a DagsHub vía API REST.

    Returns:
        (dagshub_url: str, size_mb: float) si exitoso, (None, 0.0) si falla.
    """
    if not DAGSHUB_TOKEN:
        logger.error("No se puede subir champion: falta DAGSHUB_USER_TOKEN")
        return None, 0.0

    headers = {'Authorization': f'token {DAGSHUB_TOKEN}'}
    try:
        # 1. Obtener SHA del último commit de main
        branch_url = (
            f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}"
            f"/{DAGSHUB_REPO_NAME}/branches/main"
        )
        branch_resp = requests.get(branch_url, headers=headers, timeout=15)
        if branch_resp.status_code != 200:
            logger.error(f"No se pudo obtener last_commit: HTTP {branch_resp.status_code}")
            return None, 0.0
        last_commit = branch_resp.json().get('commit', {}).get('id', '')

        # 2. Preparar upload
        model_dir      = os.path.dirname(DAGSHUB_MODEL_PATH)   # "modelos/fuga"
        model_filename = os.path.basename(DAGSHUB_MODEL_PATH)   # "modelo.pkl"
        upload_url = (
            f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}"
            f"/{DAGSHUB_REPO_NAME}/content/main/{model_dir}"
        )
        files = {'files': (model_filename, model_bytes, 'application/octet-stream')}
        data  = {
            'commit_summary': f"[Churn] Auto-Update Champion: {version_tag}",
            'commit_choice': 'direct',
            'versioning':    'git',
            'last_commit':   last_commit,
        }

        response = requests.put(upload_url, files=files, data=data, headers=headers, timeout=120)
        if response.status_code in (200, 201):
            public_url = (
                f"https://dagshub.com/{DAGSHUB_REPO_OWNER}"
                f"/{DAGSHUB_REPO_NAME}/src/main/{DAGSHUB_MODEL_PATH}"
            )
            size_mb = len(model_bytes) / (1024 * 1024)
            logger.info(f"Champion churn subido a DagsHub: {public_url}")
            return public_url, size_mb
        else:
            logger.error(f"DagsHub retorno HTTP {response.status_code}: {response.text[:300]}")
            return None, 0.0
    except Exception as e:
        logger.error(f"Error subiendo champion churn: {e}")
        return None, 0.0


def verify_champion_integrity(expected_version: str) -> bool:
    """Verifica que el modelo recién subido sea deserializable y tenga la version correcta."""
    if not DAGSHUB_TOKEN:
        return False

    import time
    headers = {'Authorization': f'token {DAGSHUB_TOKEN}'}
    url = (
        f"https://dagshub.com/{DAGSHUB_REPO_OWNER}"
        f"/{DAGSHUB_REPO_NAME}/raw/main/{DAGSHUB_MODEL_PATH}"
    )
    try:
        time.sleep(3)
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            logger.error(f"Verificacion fallo: HTTP {resp.status_code}")
            return False
        pack = joblib.load(io.BytesIO(resp.content))
        if not isinstance(pack, dict):
            logger.error("Verificacion fallo: el archivo no es un dict")
            return False
        actual = pack.get('meta_info', {}).get('version', 'UNKNOWN')
        if actual != expected_version:
            logger.error(f"Verificacion fallo: esperado={expected_version}, obtenido={actual}")
            return False
        logger.info(f"Integridad verificada: version={actual}")
        return True
    except Exception as e:
        logger.error(f"Error verificando integridad: {e}")
        return False


def notify_hot_reload(internal_token: str = "") -> bool:
    """
    Notifica al servidor principal (puerto 8000) para recargar el modelo de churn.
    Best-effort: no bloquea el flujo si falla.
    """
    reload_url = os.getenv("CHURN_RELOAD_URL", "http://localhost:8000/churn/reload")
    token = internal_token or os.getenv("INTERNAL_API_TOKEN", "")
    headers = {"X-Internal-Token": token}
    try:
        resp = requests.post(reload_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            logger.info("Hot-reload notificado al servidor principal (churn)")
            return True
        else:
            logger.warning(f"Hot-reload retorno HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        logger.warning(f"Hot-reload no disponible: {e}")
        return False
