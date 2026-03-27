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
from retiro_atm.router import router as retiro_atm_router

# Crear app FastAPI
app = FastAPI(
    title="BankMind Self-Training API - Modulo ATM",
    description="API de auto-retraining para los modelos predictivos de retiro de efectivo en cajeros automaticos",
    version="1.0.0"
)

# Registrar routers
app.include_router(retiro_atm_router)

@app.get("/health",tags=["Verificación de la disponibilidad de la api"])
async def health():
    """
    Endpoint para verificar si esta funcionando la API
    """
    return {"status": "ok", "service": "self-training-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)