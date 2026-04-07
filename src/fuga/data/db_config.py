# src/fuga/data/db_config.py
import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

load_dotenv()

logger = logging.getLogger(__name__)


def get_db_connection():
    """Conexión psycopg2 para queries raw SQL (extracción de datos de entrenamiento)."""
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "5432")
    database = os.getenv("DB_NAME",     "BankMindBetta_V3")
    user     = os.getenv("DB_USER",     "postgres")
    password = os.getenv("DB_PASSWORD", "1234")

    dsn = f"host={host} port={port} dbname={database} user={user} password={password}"
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    logger.info("Conexion psycopg2 establecida (fuga)")
    return conn


def get_db_url():
    return (
        f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
        f"{os.getenv('DB_PASSWORD', '1234')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'BankMindBetta_V3')}"
    )


engine = create_engine(
    get_db_url(),
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Session:
    """Context manager con commit/rollback automático."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error en sesion de BD (fuga): {e}")
        raise
    finally:
        session.close()
