"""
Centralized logging configuration.

Provides structured JSON logging for production and human-readable
output for local development. Import ``setup_logging`` early in the
application lifecycle (e.g. in ``main.py`` or ``worker.py``).
"""

from __future__ import annotations

import logging
import sys


def setup_logging(
    level: str = "INFO",
    *,
    json_format: bool = False,
) -> None:
    """
    Configure the root logger for the application.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        json_format: If ``True``, emit structured JSON lines.
            Recommended for containerised / production environments.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if json_format:
        fmt = (
            '{"time":"%(asctime)s",'
            '"level":"%(levelname)s",'
            '"logger":"%(name)s",'
            '"message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(log_level)

    # Avoid duplicate handlers on repeated calls
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    _silence_noisy_loggers(log_level)


def _silence_noisy_loggers(app_level: int) -> None:
    """
    Reduce verbosity of third-party libraries.

    Args:
        app_level: The application's configured log level.
    """
    noisy = [
        "urllib3",
        "httpcore",
        "httpx",
        "celery.redirected",
        "celery.worker.strategy",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(max(app_level, logging.WARNING))
