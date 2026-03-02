# src/retiro_atm/database.py
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# Credenciales de BD (matching Java backend)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "1234")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "BankMindDB")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = None
SessionLocal = None
Base = declarative_base()


def init_db():
    """Inicializa el engine y la sesión de SQLAlchemy."""
    global engine, SessionLocal
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info(f"✅ Conexión a BD configurada: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    except Exception as e:
        logger.error(f"❌ Error configurando conexión a BD: {e}")
        raise e


def get_session():
    """Crea y retorna una sesión de BD. El llamador es responsable de cerrarla."""
    if SessionLocal is None:
        init_db()
    return SessionLocal()
