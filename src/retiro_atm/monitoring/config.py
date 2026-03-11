import os
from dotenv import load_dotenv

load_dotenv()  # Carga .env si existe

# ── API de predicción ────────────────────────────────────────────
PREDICTION_API_URL = os.getenv(
    "PREDICTION_API_URL",
    "http://localhost:8000/api/atm/v1/withdrawal/off-time"
)

# ── Umbrales PSI ─────────────────────────────────────────────────
PSI_CRITICAL_THRESHOLD    = float(os.getenv("PSI_CRITICAL_THRESHOLD",    "0.20"))
PSI_WARNING_THRESHOLD     = float(os.getenv("PSI_WARNING_THRESHOLD",     "0.10"))
PSI_CRITICAL_PCT_FORCE    = float(os.getenv("PSI_CRITICAL_PCT_FORCE",    "0.50"))
PSI_WARNING_PCT_ALARM     = float(os.getenv("PSI_WARNING_PCT_ALARM",     "0.30"))

