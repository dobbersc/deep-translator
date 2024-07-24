import logging.config
import os
from pathlib import Path
from typing import Final

__all__ = ["__version__", "CACHE_DIRECTORY", "LOG_SEPARATOR"]

__version__: Final[str] = "0.0.1"

CACHE_DIRECTORY: Path = Path(os.getenv("DEEP_TRANSLATOR_CACHE") or Path.home() / ".cache" / "deep_translator")

LOG_SEPARATOR: Final[str] = "-" * 100

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {"translator": {"handlers": ["console"], "level": "INFO", "propagate": False}},
    },
)
