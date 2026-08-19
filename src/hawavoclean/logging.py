"""Structured logging for HawaVoClean."""

import logging
import sys
from typing import TextIO


def setup_logging(
    level: int = logging.INFO,
    stream: TextIO | None = None,
    log_format: str | None = None,
) -> logging.Logger:
    """Configure root HawaVoClean logger."""
    logger = logging.getLogger("hawavoclean")
    logger.setLevel(level)
    logger.handlers.clear()

    if log_format is None:
        log_format = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"

    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger(name: str = "hawavoclean") -> logging.Logger:
    """Get a child logger under the hawavoclean namespace."""
    return logging.getLogger(f"hawavoclean.{name}" if name != "hawavoclean" else "hawavoclean")
