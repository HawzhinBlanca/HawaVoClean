"""Structured logging for Hawzhin VoiceClean."""

import logging
import sys
from typing import TextIO


def setup_logging(
    level: int = logging.INFO,
    stream: TextIO | None = None,
    log_format: str | None = None,
) -> logging.Logger:
    """Configure root VoiceClean logger."""
    logger = logging.getLogger("voiceclean")
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


def get_logger(name: str = "voiceclean") -> logging.Logger:
    """Get a child logger under the voiceclean namespace."""
    return logging.getLogger(f"voiceclean.{name}" if name != "voiceclean" else "voiceclean")
