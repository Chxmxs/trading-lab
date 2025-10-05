# -*- coding: utf-8 -*-
"""
Logging configuration for the Trading-Lab project.

Provides:
- configure_logging(name=None, level=None, logfile=None)
    Returns a configured logger with both console + rotating file handlers.
    Safe to call multiple times (no duplicate handlers).
- LOG_DIR: Path to the logs directory under companion/

Environment variables:
- TL_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default: INFO)
- TL_LOG_FILE: override log file path if desired

This module is intentionally standalone and UTF-8 clean.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Optional

# Logs directory (under companion/)
LOG_DIR = Path("companion") / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Default log file (you can override via TL_LOG_FILE)
DEFAULT_LOG_FILE = LOG_DIR / "trading_lab.log"

# Simple, readable formatter
_FORMAT = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _level_from_env(default: str = "INFO") -> int:
    env = os.getenv("TL_LOG_LEVEL", default).upper()
    return getattr(logging, env, logging.INFO)


def _ensure_file_handler(logfile: Path) -> RotatingFileHandler:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=str(logfile),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",  # ensure UTF-8 log files
        delay=True,
    )
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    return handler


def _ensure_console_handler() -> logging.Handler:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    return ch


def configure_logging(
    name: Optional[str] = None,
    level: Optional[int] = None,
    logfile: Optional[Path | str] = None,
) -> logging.Logger:
    """
    Create or return a logger configured with console + rotating file handlers.

    Args:
        name: logger name (module usually). If None, uses root logger.
        level: logging level; if None, derives from TL_LOG_LEVEL env (default INFO).
        logfile: path to log file; if None, uses TL_LOG_FILE env or DEFAULT_LOG_FILE.

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name) if name else logging.getLogger()
    # Derive level
    eff_level = level if level is not None else _level_from_env()
    logger.setLevel(eff_level)

    # Decide logfile
    lf_env = os.getenv("TL_LOG_FILE")
    logfile_path = Path(lf_env) if lf_env else (Path(logfile) if logfile else DEFAULT_LOG_FILE)

    # Avoid duplicate handlers if already configured
    have_file = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
    have_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
                       for h in logger.handlers)

    if not have_console:
        logger.addHandler(_ensure_console_handler())
    if not have_file:
        logger.addHandler(_ensure_file_handler(logfile_path))

    # Reduce noisy libs if using root
    if not name:
        for noisy in ("matplotlib", "urllib3", "mlflow", "botocore"):
            logging.getLogger(noisy).setLevel(max(eff_level, logging.WARNING))

    return logger


# Backwards-compat aliases (if your older code used different names)
get_logger = configure_logging
setup_logging = configure_logging
