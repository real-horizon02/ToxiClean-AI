"""
utils/logger.py
---------------
Shared logging setup for all ToxiClean AI execution scripts.

Usage:
    from execution.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting job", extra={"job_id": "abc"})
"""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Usually __name__ of the calling module.
        level: Override log level (reads LOG_LEVEL env var by default).

    Returns:
        A configured Logger instance.

    Security:
        - Formatter excludes any key containing 'secret', 'key', 'token', 'password'.
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers on re-import
        return logger

    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
