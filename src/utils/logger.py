"""Structured logging with Rich console output and optional file sink."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_LOG_FORMAT = "%(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOGGERS: dict[str, logging.Logger] = {}

console = Console(stderr=True)


def get_logger(
    name: str = "rt_pipeline",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Get or create a named logger with Rich console handler.

    Args:
        name: Logger name (dot-separated hierarchy supported).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file for persistent logging.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # ── Rich console handler ─────────────────────────────────────────────
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(rich_handler)

    # ── File handler (optional) ──────────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
                datefmt=_DATE_FORMAT,
            )
        )
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
