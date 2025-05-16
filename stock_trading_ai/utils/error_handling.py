"""Error handling utilities for stock trading AI."""

import logging
import functools
import time
from typing import Callable, Any, TypeVar, Optional, Tuple, Type, Union
import requests
from requests.exceptions import RequestException
import yfinance as yf
import finnhub

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

def handle_finnhub_errors(func: F) -> F:
    """Decorator to handle Finnhub-specific errors.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except finnhub.exceptions.FinnhubAPIException as e:
            raise DataFetchError(f"Finnhub API error: {str(e)}")
        except Exception as e:
            raise DataFetchError(f"Finnhub error: {str(e)}")

    return wrapper  # type: ignore

def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[F], F]:
    """Decorator to retry a function on exception.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exception  # type: ignore
        return wrapper  # type: ignore
    return decorator

def cleanup_on_error(func: F) -> F:
    """Decorator to perform cleanup on error.
    
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
            logger.error(f"Error in {func.__name__}: {str(e)}")
            # Add cleanup logic here
            raise
    return wrapper  # type: ignore

class ErrorHandler:
    """Error handler for stock trading AI."""
    
    @staticmethod
    def handle_api_error(error: Exception) -> None:
        """Handle API errors.
        
        Args:
            error: Exception to handle
        """
        if isinstance(error, finnhub.exceptions.FinnhubAPIException):
            logger.error(f"Finnhub API error: {str(error)}")
        elif isinstance(error, RequestException):
            logger.error(f"Request error: {str(error)}")
        else:
            logger.error(f"Unknown error: {str(error)}")

def safe_api_call(func: F) -> F:
    """Decorator to safely make API calls.
    
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
            ErrorHandler.handle_api_error(e)
            raise DataFetchError(f"API call failed: {str(e)}")
    return wrapper  # type: ignore

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