# main.py
import sys
import os
from fastapi import FastAPI

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

# Herramientas de log y monitoreo
from src.configuration.logging_config import setup_logging

# Iniciamos el logging
setup_logging()

import logging
logger = logging.getLogger(__name__)

# Importar routers
from morosidad.router import router as morosidad_router
from fraude.router import router as fraude_router

# Crear app FastAPI
app = FastAPI(
    title="BankMind Self-Training API",
    description="API de auto-retraining para los modelos de BankMind",
    version="1.0.0"
)

# Registrar routers
app.include_router(morosidad_router)
app.include_router(fraude_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "self-training-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
