"""logging_utils.py — Rich-formatted project logger."""

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            markup=True,
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
