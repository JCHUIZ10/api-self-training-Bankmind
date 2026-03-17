# src/retiro_atm/database.py
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()  # Carga .env si existe

# Credenciales de BD (matching Java backend)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Pool de conexiones (para uso en segundo plano)
DB_POOL_SIZE        = int(os.getenv("DB_POOL_SIZE",    "5"))
DB_MAX_OVERFLOW     = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT     = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE     = int(os.getenv("DB_POOL_RECYCLE", "1800"))  # 30 min

engine = None
SessionLocal = None
Base = declarative_base()

def init_db():
    """Inicializa el engine y la sesión de SQLAlchemy."""
    global engine, SessionLocal
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            pool_recycle=DB_POOL_RECYCLE,   # Evita conexiones obsoletas
            pool_pre_ping=True,             # Verifica conexión antes de usarla
            echo=False,
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info(f"✅ Conexión a BD configurada: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    except Exception as e:
        logger.error(f"❌ Error configurando conexión a BD: {e}")
        raise e

def get_session():
    """Crea y retorna una sesión de BD. El llamador es responsable de cerrarla."""
    if SessionLocal is None:
        init_db()
    return SessionLocal() # type: ignore

def get_engine():
    global engine
    if engine is None:
        init_db()
    return engine