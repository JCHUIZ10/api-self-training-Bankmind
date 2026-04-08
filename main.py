# main.py
import sys
import os
from fastapi import FastAPI

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

# Herramientas de log y monitoreo
from src.configuration.logging_config import setup_logging
from dotenv import load_dotenv

load_dotenv()
# Iniciamos el logging
setup_logging()

import logging
logger = logging.getLogger(__name__)

# Importar routers
from morosidad.router import router as morosidad_router
from fraude.api.router import router as fraude_router
from retiro_atm.router import router as retiro_atm_router
from fuga.api.router import router as fuga_router

# Importar scheduler de monitor
from fuga.service.monitor_scheduler import setup_churn_monitor, shutdown_churn_monitor

# Crear app FastAPI
app = FastAPI(
    title="BankMind Self-Training API",
    description="API de auto-retraining para los modelos de BankMind",
    version="1.0.0"
)

# Registrar routers
app.include_router(morosidad_router)
app.include_router(fraude_router)
app.include_router(retiro_atm_router)
app.include_router(fuga_router)


@app.on_event("startup")
async def startup_event():
    logger.info("[STARTUP] Iniciando monitor de rendimiento Churn...")
    setup_churn_monitor()


@app.on_event("shutdown")
async def shutdown_event():
    shutdown_churn_monitor()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "self-training-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)