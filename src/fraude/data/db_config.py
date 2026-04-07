# src/fraude/db_config.py
import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")


def get_db_connection():
    """
    Crea una conexión a la base de datos PostgreSQL usando psycopg2.
    Lee las credenciales desde variables de entorno.
    Usado para queries raw SQL (ej. data extraction).
    """
    try:
        # Usar DSN string con las variables de entorno
        dsn = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
        
        conn = psycopg2.connect(
            dsn,
            cursor_factory=RealDictCursor
        )
        logger.info("✅ Conexión a base de datos establecida (psycopg2)")
        return conn
    except Exception as e:
        logger.error(f"❌ Error conectando a la base de datos: {e}")
        raise


# ========================================
# SQLAlchemy Configuration
# ========================================

def get_db_url():
    """Construye la URL de conexión para SQLAlchemy"""
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Engine global (creado una vez)
engine = create_engine(
    get_db_url(),
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True  # Verifica conexión antes de usar
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Session:
    """
    Context manager para obtener una sesión de SQLAlchemy.
    Maneja automáticamente commit/rollback.
    
    Uso:
        with get_db_session() as session:
            result = model_registry.save_model_metadata(session, ...)
            session.commit()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Error en sesión de BD: {e}")
        raise
    finally:
        session.close()