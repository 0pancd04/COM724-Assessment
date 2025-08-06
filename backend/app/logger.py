# backend/app/logger.py
import os
import logging
import inspect
import functools
from logging.handlers import RotatingFileHandler
from typing import Optional, Any, Callable


class EnhancedLogger:
    """Enhanced logger that automatically includes file and function context"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _get_caller_info(self, skip_frames: int = 2) -> tuple:
        """Get caller file and function name"""
        try:
            frame = inspect.currentframe()
            for _ in range(skip_frames):
                if frame is not None:
                    frame = frame.f_back
            
            if frame is not None:
                filename = os.path.basename(frame.f_code.co_filename)
                function_name = frame.f_code.co_name
                line_number = frame.f_lineno
                return filename, function_name, line_number
        except Exception:
            pass
        return "unknown.py", "unknown_function", 0
    
    def _format_message(self, message: str, extra_context: Optional[str] = None) -> str:
        """Format message with caller context"""
        filename, function_name, line_number = self._get_caller_info(skip_frames=3)
        context = f"[{filename}:{function_name}:{line_number}]"
        if extra_context:
            context = f"[{extra_context}] {context}"
        return f"{context} {message}"
    
    def debug(self, message: str, extra_context: Optional[str] = None):
        """Log debug message with context"""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.debug(formatted_msg)
    
    def info(self, message: str, extra_context: Optional[str] = None):
        """Log info message with context"""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.info(formatted_msg)
    
    def warning(self, message: str, extra_context: Optional[str] = None):
        """Log warning message with context"""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.warning(formatted_msg)
    
    def error(self, message: str, extra_context: Optional[str] = None):
        """Log error message with context"""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.error(formatted_msg)
    
    def critical(self, message: str, extra_context: Optional[str] = None):
        """Log critical message with context"""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.critical(formatted_msg)
    
    def exception(self, message: str, extra_context: Optional[str] = None):
        """Log exception message with context"""
        formatted_msg = self._format_message(message, extra_context)
        self.logger.exception(formatted_msg)


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


def setup_enhanced_logger(
    logger_name: str,
    log_file_path: str,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3
) -> EnhancedLogger:
    """
    Configure and return an enhanced logger with automatic context detection.
    
    Args:
        logger_name: name for the logger instance.
        log_file_path: full path to the log file (directories will be created as needed).
        level: logging level.
        max_bytes: max bytes per file before rotation.
        backup_count: number of backup files to keep.
    """
    base_logger = setup_logger(logger_name, log_file_path, level, max_bytes, backup_count)
    return EnhancedLogger(base_logger)


def log_function_entry_exit(logger: EnhancedLogger, log_args: bool = False):
    """
    Decorator to automatically log function entry and exit
    
    Args:
        logger: EnhancedLogger instance
        log_args: Whether to log function arguments
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Log entry
            if log_args:
                args_str = f"args={args}, kwargs={kwargs}"
                logger.debug(f"Entering {func_name} with {args_str}")
            else:
                logger.debug(f"Entering {func_name}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Exception in {func_name}: {str(e)}")
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = func.__name__
            
            # Log entry
            if log_args:
                args_str = f"args={args}, kwargs={kwargs}"
                logger.debug(f"Entering async {func_name} with {args_str}")
            else:
                logger.debug(f"Entering async {func_name}")
            
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Exiting async {func_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Exception in async {func_name}: {str(e)}")
                raise
        
        return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
    return decorator
