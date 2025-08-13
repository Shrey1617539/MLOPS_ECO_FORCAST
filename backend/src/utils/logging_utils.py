import logging
from functools import wraps
from typing import Callable, Optional

# Dictionary to store logger instances
loggers = {}

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name of the logger (usually the module name)
    
    Returns:
        Logger instance
    """
    global loggers
    
    if name in loggers:
        return loggers[name]
    
    logger = logging.getLogger(name)
    loggers[name] = logger
    
    return logger

def log_function_call(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator to log function calls, arguments, and return values.
    
    Args:
        logger: Logger to use. If None, a logger will be created based on the module name.
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.exception(f"Exception in {func_name}: {str(e)}")
                raise
        
        return wrapper
    
    return decorator
