# src/retiro_atm/dagshub_client.py
import io
import os
import logging
import time

import joblib
import requests
import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuración DagsHub desde variables de entorno
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "notificacionesbankmind")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "Modelos_BankMind_2026")
DAGSHUB_MODEL_PATH = os.getenv("DAGSHUB_ATM_MODEL_PATH", "modelos/retiro-atm/modelo.pkl")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")


class AtmModelProvider:
    """
    Singleton para gestionar modelos de retiro ATM en DagsHub.
    Maneja descarga, upload y verificación de modelos.
    """

    _instance = None
    _dagshub_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._modelo_cache = None
        return cls._instance

    @classmethod
    def init_dagshub_connection(cls):
        """Inicializa la conexión a DagsHub y MLflow tracking."""
        if cls._dagshub_initialized:
            return

        if not DAGSHUB_TOKEN:
            logger.warning("⚠️ DAGSHUB_USER_TOKEN no configurado. MLflow remoto no funcionará.")
            return

        try:
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True,
            )
            cls._dagshub_initialized = True
            logger.info(f"✅ DagsHub conectado: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
            logger.info(f"📡 MLflow URI: {mlflow.get_tracking_uri()}")
        except Exception as e:
            logger.error(f"❌ Error inicializando DagsHub: {e}")

    def obtener_modelo_produccion(self, force_download: bool = False):
        """
        Descarga el modelo de producción desde DagsHub.

        Args:
            force_download: Si True, ignora el caché.

        Returns:
            Modelo XGBRegressor o None si no existe.
        """
        if self._modelo_cache is not None and not force_download:
            logger.info("📦 Usando modelo champion cacheado")
            return self._modelo_cache

        if not DAGSHUB_TOKEN:
            logger.warning("No se puede descargar champion: falta DAGSHUB_USER_TOKEN")
            return None

        headers = {"Authorization": f"token {DAGSHUB_TOKEN}"}

        for branch in ("main", "master"):
            url = (
                f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
                f"/raw/{branch}/{DAGSHUB_MODEL_PATH}"
            )
            logger.info(f"⬇️ Intentando descargar champion desde: {url}")

            try:
                response = requests.get(url, headers=headers, timeout=60)
                if response.status_code == 200:
                    model_pack = joblib.load(io.BytesIO(response.content))

                    if isinstance(model_pack, dict):
                        self._modelo_cache = model_pack.get("modelo_prediccion")
                    else:
                        self._modelo_cache = model_pack

                    logger.info("✅ Champion descargado exitosamente")
                    return self._modelo_cache
                else:
                    logger.warning(f"Rama '{branch}': HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"Error descargando de {branch}: {e}")

        logger.warning("⚠️ No se encontró modelo champion en DagsHub (Cold Start?)")
        return None

    @staticmethod
    def actualizar_modelo_produccion(nuevo_modelo, version_tag: str) -> bool:
        """
        Empaqueta y sube el nuevo champion a DagsHub.

        Args:
            nuevo_modelo: Modelo XGBRegressor entrenado.
            version_tag: Etiqueta de versión (e.g. "v3").

        Returns:
            True si el upload fue exitoso.
        """
        if not DAGSHUB_TOKEN:
            logger.error("❌ No se puede subir champion: falta DAGSHUB_USER_TOKEN")
            return False

        try:
            # Empaquetar modelo con metadatos
            model_pack = {
                "modelo_prediccion": nuevo_modelo,
                "meta_info": {"version": version_tag},
            }

            buffer = io.BytesIO()
            joblib.dump(model_pack, buffer)
            model_bytes = buffer.getvalue()

            headers = {"Authorization": f"token {DAGSHUB_TOKEN}"}

            # Obtener último commit SHA
            branch_url = (
                f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/"
                f"{DAGSHUB_REPO_NAME}/branches/main"
            )
            branch_resp = requests.get(branch_url, headers=headers, timeout=15)
            if branch_resp.status_code != 200:
                logger.error(f"❌ No se pudo obtener last_commit: HTTP {branch_resp.status_code}")
                return False

            last_commit = branch_resp.json().get("commit", {}).get("id", "")
            logger.info(f"📌 Last commit: {last_commit[:12]}...")

            # Preparar upload
            model_dir = os.path.dirname(DAGSHUB_MODEL_PATH)
            model_filename = os.path.basename(DAGSHUB_MODEL_PATH)

            upload_url = (
                f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/"
                f"{DAGSHUB_REPO_NAME}/content/main/{model_dir}"
            )

            files = {"files": (model_filename, model_bytes, "application/octet-stream")}
            data = {
                "commit_summary": f"Auto-Update Champion ATM: {version_tag}",
                "commit_message": f"Modelo retiro-ATM actualizado. Versión: {version_tag}",
                "commit_choice": "direct",
                "versioning": "dvc",
                "last_commit": last_commit,
            }

            response = requests.put(
                upload_url, files=files, data=data,
                headers=headers, timeout=120,
            )

            if response.status_code in (200, 201):
                logger.info(f"✅ Champion subido a DagsHub (HTTP {response.status_code})")
                return True
            else:
                logger.error(
                    f"❌ DagsHub retornó HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Error subiendo champion: {e}")
            return False

    @staticmethod
    def verificar_integridad(expected_version: str) -> bool:
        """
        Verifica la integridad re-descargando el modelo y comparando versión.

        Returns:
            True si el archivo existe, se deserializa, y la versión coincide.
        """
        if not DAGSHUB_TOKEN:
            logger.error("❌ No se puede verificar: falta DAGSHUB_USER_TOKEN")
            return False

        headers = {"Authorization": f"token {DAGSHUB_TOKEN}"}
        url = (
            f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
            f"/raw/main/{DAGSHUB_MODEL_PATH}"
        )

        try:
            time.sleep(3)  # Esperar propagación del commit
            response = requests.get(url, headers=headers, timeout=60)

            if response.status_code != 200:
                logger.error(f"❌ Verificación falló: HTTP {response.status_code}")
                return False

            model_pack = joblib.load(io.BytesIO(response.content))

            if not isinstance(model_pack, dict):
                logger.error("❌ Verificación falló: formato inesperado")
                return False

            actual_version = model_pack.get("meta_info", {}).get("version", "UNKNOWN")
            if actual_version != expected_version:
                logger.error(
                    f"❌ Versión esperada={expected_version}, obtenida={actual_version}"
                )
                return False

            has_model = model_pack.get("modelo_prediccion") is not None
            if not has_model:
                logger.error("❌ 'modelo_prediccion' no encontrado en el pack")
                return False

            logger.info(f"✅ Integridad verificada: versión={actual_version}")
            return True

        except Exception as e:
            logger.error(f"❌ Error en verificación: {e}")
            return False
