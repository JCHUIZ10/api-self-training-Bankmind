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
            "encoding": "utf-8"
        },
        "file_retiro_atm_self_train": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "retiro_atm_self_train.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
            "encoding": "utf-8"
        },
        "file_retiro_atm_monitoring": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "retiro_atm_monitoring.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
            "encoding": "utf-8"
        },
        "file_retiro_atm_generated": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "retiro_atm_generated.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
            "encoding": "utf-8"
        },
        "file_retiro_atm": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "retiro_atm.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
            "encoding": "utf-8"
        },
        "file_fraude": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "fraude.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
            "encoding": "utf-8"
        },
        "file_morosidad": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": LOG_DIR / "morosidad.log",
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
            "encoding": "utf-8"
        },
    },
    "loggers": {
        "retiro_atm.self_train": {"handlers": ["file_retiro_atm_self_train", "console"], "level": "INFO", "propagate": False},
        "retiro_atm.monitoring": {"handlers": ["file_retiro_atm_monitoring", "console"], "level": "INFO", "propagate": False},
        "retiro_atm.generated": {"handlers": ["file_retiro_atm_generated", "console"], "level": "INFO", "propagate": False},
        "retiro_atm": {"handlers": ["file_retiro_atm", "console"], "level": "INFO", "propagate": False},
        "fraude": {"handlers": ["file_fraude", "console"], "level": "INFO", "propagate": False},
        "morosidad": {"handlers": ["file_morosidad", "console"], "level": "INFO", "propagate": False},
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