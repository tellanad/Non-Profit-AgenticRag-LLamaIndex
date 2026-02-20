"""
Structured logging setup.

WHY STRUCTURED LOGGING?
- In production, you grep logs. Structured (JSON) logs are searchable.
- In development, you read logs. Pretty-printed logs are readable.
- This module gives you both, controlled by LOG_LEVEL env var.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure application-wide logging.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Root logger configured for the application
    """
    logger = logging.getLogger("agentic_rag")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on re-initialization
    if logger.handlers:
        return logger

    # Console handler with readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Usage:
        from config.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Ingesting document", extra={"doc": "policy.pdf"})
    """
    return logging.getLogger(f"agentic_rag.{name}")
