import os
from dotenv import load_dotenv

load_dotenv()  # Carga .env si existe

# ── API de predicción ────────────────────────────────────────────
PREDICTION_API_URL = os.getenv("PREDICTION_API_URL")

# ── Umbrales PSI ─────────────────────────────────────────────────
PSI_CRITICAL_THRESHOLD    = float(os.getenv("PSI_CRITICAL_THRESHOLD"))
PSI_WARNING_THRESHOLD     = float(os.getenv("PSI_WARNING_THRESHOLD"))
PSI_CRITICAL_PCT_FORCE    = float(os.getenv("PSI_CRITICAL_PCT_FORCE"))
PSI_WARNING_PCT_ALARM     = float(os.getenv("PSI_WARNING_PCT_ALARM"))

