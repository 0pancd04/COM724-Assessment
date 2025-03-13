import os
import logging

# --- Logging Setup (unchanged from before) ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
else:
    for file in os.listdir(LOG_DIR):
        file_path = os.path.join(LOG_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def setup_logger(logger_name: str, log_filename: str, level=logging.INFO, max_bytes=5 * 1024 * 1024, backup_count=3):
    log_file = os.path.join(LOG_DIR, log_filename)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    from logging.handlers import RotatingFileHandler
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger