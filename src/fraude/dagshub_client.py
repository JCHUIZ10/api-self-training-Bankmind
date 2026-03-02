import os
import logging
import io
import time
import joblib
import requests
import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuración DagsHub (Hardcoded o env vars)
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "notificacionesbankmind")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "Modelos_BankMind_2026")
DAGSHUB_MODEL_PATH = os.getenv("DAGSHUB_MODEL_PATH", "modelos/fraude/modelo.pkl")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")

_dagshub_initialized = False

def init_dagshub_connection():
    """Inicializa la conexión a DagsHub y MLflow tracking."""
    global _dagshub_initialized
    if not _dagshub_initialized:
        if not DAGSHUB_TOKEN:
            logger.warning("[WARN] DAGSHUB_USER_TOKEN no está configurado. MLflow no funcionará remotamente.")
            return

        try:
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True
            )
            _dagshub_initialized = True
            logger.info(f"[OK] Conectado a DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
            logger.info(f"[MLFLOW] MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        except Exception as e:
            logger.error(f"[ERROR] Error inicializando DagsHub: {e}")

def download_current_champion():
    """
    Descarga el modelo 'Production' actual (rama main) desde DagsHub.
    Retorna el objeto modelo (pipeline) y metadatos si existen.
    
    Returns:
        tuple: (modelo, explainer, metadata) o (None, None, None) si falla.
    """
    if not DAGSHUB_TOKEN:
        logger.warning("[WARN] No se puede descargar Champion: Falta Token")
        return None, None, None

    # Using Bearer token for better security/compatibility
    headers = {'Authorization': f'Bearer {DAGSHUB_TOKEN}'}
    # Intentar descargar de main/master
    branches = ["main", "master"]
    
    for branch in branches:
        url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/raw/{branch}/{DAGSHUB_MODEL_PATH}"
        logger.info(f"[DOWNLOAD] Intentando descargar Champion desde: {url}")
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                logger.info("[OK] Champion descargado exitosamente. Cargando en memoria...")
                model_pack = joblib.load(io.BytesIO(response.content))
                
                # Extraer componentes (formato nuevo dict o legacy object)
                if isinstance(model_pack, dict):
                    return (
                        model_pack.get('modelo_prediccion'),
                        model_pack.get('shap_explainer'),
                        model_pack.get('meta_info', {})
                    )
                else:
                    # Legacy
                    return model_pack, None, {}
            else:
                logger.warning(f"[WARN] Rama '{branch}' retornó status {response.status_code}")
        except Exception as e:
            logger.error(f"[ERROR] Error descargando de {branch}: {e}")
            
    logger.warning("[WARN] No se encontró modelo Champion en DagsHub (Cold Start?)")
    return None, None, None

def upload_champion(model_bytes: bytes, version_tag: str):
    """
    Sube el nuevo modelo Champion a DagsHub.
    Usa la API REST directa para evitar bugs del SDK (last_commit issue).
    
    Returns:
        tuple: (url_dagshub, size_mb) si exitoso, (None, 0.0) si falla.
    """
    if not DAGSHUB_TOKEN:
        logger.error("[ERROR] No se puede subir Champion: Falta Token")
        return None, 0.0

    try:
        repo_fullname = f"{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
        logger.info(f"[UPLOAD] Subiendo nuevo Champion ({version_tag}) a {repo_fullname}...")
        
        # Using Bearer token
        headers = {'Authorization': f'Bearer {DAGSHUB_TOKEN}'}
        
        # 1) Obtener último commit SHA de la rama main (requerido por DagsHub)
        branch_url = f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/branches/main"
        branch_resp = requests.get(branch_url, headers=headers, timeout=15)
        if branch_resp.status_code != 200:
            logger.error(f"[ERROR] No se pudo obtener last_commit: HTTP {branch_resp.status_code}")
            return False
        last_commit = branch_resp.json().get('commit', {}).get('id', '')
        logger.info(f"[INFO] Last commit SHA: {last_commit[:12]}...")
        
        # 2) Preparar upload
        model_dir = os.path.dirname(DAGSHUB_MODEL_PATH)      # "modelos/fraude"
        model_filename = os.path.basename(DAGSHUB_MODEL_PATH)  # "modelo.pkl"
        
        upload_url = (
            f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
            f"/content/main/{model_dir}"
        )
        
        files = {
            'files': (model_filename, model_bytes, 'application/octet-stream')
        }
        data = {
            'commit_summary': f"Auto-Update Champion: {version_tag}",
            'commit_message': f"Modelo actualizado automaticamente. Version: {version_tag}",
            'commit_choice': 'direct',
            'versioning': 'dvc',
            'last_commit': last_commit
        }
        
        response = requests.put(
            upload_url,
            files=files,
            data=data,
            headers=headers,
            timeout=120
        )
        
        if response.status_code in (200, 201):
            logger.info(f"[OK] Champion actualizado correctamente en DagsHub (HTTP {response.status_code})")
            
            # Construir URL pública del archivo
            public_url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/src/main/{DAGSHUB_MODEL_PATH}"
            size_mb = len(model_bytes) / (1024 * 1024)
            
            return public_url, size_mb
        else:
            logger.error(f"[ERROR] DagsHub retorno HTTP {response.status_code}: {response.text[:300]}")
            return None, 0.0
        
    except Exception as e:
        logger.error(f"[ERROR] Error subiendo Champion: {e}")
        return None, 0.0


def verify_champion_integrity(expected_version: str) -> bool:
    """
    Verifica la integridad del upload re-descargando el .pkl desde DagsHub
    y comparando la versión embebida en meta_info con la esperada.
    
    Args:
        expected_version: Version tag que se acaba de subir (ej: 'v_1739700000')
    
    Returns:
        True si el archivo existe, se puede deserializar, y la versión coincide.
    """
    logger.info(f"[INFO] Verificando integridad del upload en DagsHub (esperado: {expected_version})...")
    
    if not DAGSHUB_TOKEN:
        logger.error("[ERROR] No se puede verificar: Falta Token")
        return False

    # Using Bearer token
    headers = {'Authorization': f'Bearer {DAGSHUB_TOKEN}'}
    
    # Intentar descargar desde main (donde recién se subió)
    url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/raw/main/{DAGSHUB_MODEL_PATH}"
    
    try:
        import time as _time
        _time.sleep(3)  # Espera para que DagsHub registre el commit
        
        response = requests.get(url, headers=headers, timeout=60)
        
        if response.status_code != 200:
            logger.error(f"[ERROR] Verificación falló: DagsHub retornó status {response.status_code}")
            return False
        
        # Deserializar y verificar versión
        model_pack = joblib.load(io.BytesIO(response.content))
        
        if not isinstance(model_pack, dict):
            logger.error("[ERROR] Verificación falló: El archivo descargado no es un dict (formato inesperado)")
            return False

        actual_version = model_pack.get('meta_info', {}).get('version', 'UNKNOWN')
        
        if actual_version != expected_version:
            logger.error(f"[ERROR] Verificación falló: Versión esperada={expected_version}, obtenida={actual_version}")
            return False
        
        # Verificar que el modelo y SHAP están presentes
        has_model = model_pack.get('modelo_prediccion') is not None
        has_shap = model_pack.get('shap_explainer') is not None
        
        if not has_model:
            logger.error("[ERROR] Verificación falló: No se encontró 'modelo_prediccion' en el pack")
            return False
        
        logger.info(f"[OK] Integridad verificada: versión={actual_version}, modelo={'OK' if has_model else 'FALTA'}, SHAP={'OK' if has_shap else 'FALTA (opcional)'}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Error en verificación de integridad: {e}")
        return False
