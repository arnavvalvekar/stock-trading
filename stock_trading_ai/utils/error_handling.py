"""Error handling utilities for stock trading AI."""

import logging
import functools
import time
from typing import Callable, Any, TypeVar, Optional, Tuple, Type, Union
import requests
from requests.exceptions import RequestException
import yfinance as yf

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for decorator typing
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class StockAIError(Exception):
    """Base exception for Stock AI system."""
    pass

class DataFetchError(StockAIError):
    """Error raised when data fetching fails."""
    pass

class ModelError(StockAIError):
    """Error raised when model operations fail."""
    pass

class ConfigurationError(StockAIError):
    """Error raised when configuration is invalid."""
    pass

def retry_on_exception(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Optional[Tuple[Type[Exception], ...]] = None
) -> Callable[[F], F]:
    """Decorator to retry a function on exception.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch
    
    Returns:
        Decorated function
    """
    if exceptions is None:
        exceptions = (Exception,)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    func_name = getattr(func, '__name__', str(func))
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func_name}: {str(e)}. "
                        f"{'Retrying in ' + str(current_delay) + 's' if attempt < max_attempts - 1 else 'Max attempts reached'}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= 2  # Exponential backoff

            if last_exception is not None:
                raise last_exception

            return None  # Should never reach here

        return wrapper  # type: ignore

    return decorator

def handle_yfinance_errors(func: F) -> F:
    """Decorator to handle yfinance-specific errors.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise DataFetchError(f"YFinance error: {str(e)}")

    return wrapper  # type: ignore

def cleanup_on_error(func: F) -> F:
    """Decorator to ensure cleanup on error.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if hasattr(args[0], 'cleanup'):
                args[0].cleanup()
            raise

    return wrapper  # type: ignore

class ErrorHandler:
    """Context manager for error handling."""

    def __init__(self, cleanup_func: Optional[Callable[[], None]] = None):
        """Initialize error handler.
        
        Args:
            cleanup_func: Optional cleanup function to call on error
        """
        self.cleanup_func = cleanup_func
        self.original_error = None

    def __enter__(self) -> 'ErrorHandler':
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], 
                 exc_val: Optional[BaseException], 
                 exc_tb: Optional[Any]) -> bool:
        if exc_type is not None:
            self.original_error = exc_val
            if self.cleanup_func is not None:
                try:
                    self.cleanup_func()
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed: {str(cleanup_error)}")
                    # Re-raise the original error
                    if self.original_error is not None:
                        raise self.original_error
        return False

def safe_api_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for making safe API calls with retries and error handling.
    
    Args:
        func: API function to call
    
    Returns:
        Decorated function
    """
    @retry_on_exception(max_attempts=3, exceptions=(RequestException,))
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except RequestException as e:
            raise DataFetchError(f"API call failed: {str(e)}")

    return wrapper 