import logging
from logging.config import dictConfig
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Formato común
DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": DEFAULT_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file_app": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "app.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
        },
        "file_fraude": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "fraude.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
        },
    },
    "loggers": {
        "fraude": {"handlers": ["file_fraude", "console"], "level": "INFO", "propagate": False},
        "uvicorn": {"level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"level": "INFO"},
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file_app"],
    },
}

def setup_logging():
    dictConfig(LOGGING_CONFIG)