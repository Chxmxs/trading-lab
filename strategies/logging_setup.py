from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time

def get_logger(name: str, file_path: str) -> logging.Logger:
    log_dir = Path(file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt.converter = time.gmtime  # UTC

    fh = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
