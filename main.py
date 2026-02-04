# main.py
import sys
import os
from fastapi import BackgroundTasks, FastAPI, HTTPException

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

# Herramientas de log y monitoreo
from src.configuration.logging_config import setup_logging

#Iniciamos el loggin
setup_logging()

## Colocar antes de la definicion de la clase : logger = logging.getLogger(__name__)
