"""
logging_utils.py
────────────────
Minimal logging utility for ToolFormer scripts.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger instance with stdout handler.

    Creates a logger with a StreamHandler writing to stdout using a
    standard format. If the logger already has handlers, skips adding
    a new one to avoid duplicate output.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level (default: logging.INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
