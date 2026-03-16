"""Logging helpers for FoldPy."""

from __future__ import annotations

import logging


def get_logger(name: str = "foldpy") -> logging.Logger:
    """Return a package logger with a safe default handler.

    Args:
        name: Logger name.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
