# backend/app/logger.py
import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(
    logger_name: str,
    log_file_path: str,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3
) -> logging.Logger:
    """
    Configure and return a logger that writes to the specified file path.

    Args:
        logger_name: name for the logger instance.
        log_file_path: full path to the log file (directories will be created as needed).
        level: logging level.
        max_bytes: max bytes per file before rotation.
        backup_count: number of backup files to keep.
    """
    # Ensure directory exists
    log_path = os.path.abspath(log_file_path)
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Remove old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Rotating file handler
    handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
