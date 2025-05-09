"""Tests for error handling utilities."""

import pytest
import requests
from unittest.mock import MagicMock, patch
from stock_trading_ai.utils.error_handling import (
    retry_on_exception,
    handle_yfinance_errors,
    cleanup_on_error,
    ErrorHandler,
    safe_api_call,
    DataFetchError
)

def test_retry_on_exception_success():
    """Test retry decorator with successful execution."""
    mock_func = MagicMock(return_value="success")
    decorated_func = retry_on_exception()(mock_func)
    
    result = decorated_func()
    assert result == "success"
    assert mock_func.call_count == 1

def test_retry_on_exception_failure():
    """Test retry decorator with failed execution."""
    mock_func = MagicMock(side_effect=ValueError("test error"))
    decorated_func = retry_on_exception(max_attempts=3, delay=0.1)(mock_func)
    
    with pytest.raises(ValueError):
        decorated_func()
    assert mock_func.call_count == 3

def test_retry_on_exception_specific_exception():
    """Test retry decorator with specific exception."""
    mock_func = MagicMock(side_effect=ValueError("test error"))
    decorated_func = retry_on_exception(
        max_attempts=3,
        delay=0.1,
        exceptions=(ValueError,)
    )(mock_func)
    
    with pytest.raises(ValueError):
        decorated_func()
    assert mock_func.call_count == 3

def test_handle_yfinance_errors():
    """Test YFinance error handling."""
    mock_func = MagicMock(side_effect=Exception("YFinance error"))
    decorated_func = handle_yfinance_errors(mock_func)
    
    with pytest.raises(DataFetchError):
        decorated_func()

def test_cleanup_on_error():
    """Test cleanup on error."""
    class TestClass:
        def __init__(self):
            self.cleaned_up = False
        
        def cleanup(self):
            self.cleaned_up = True
        
        @cleanup_on_error
        def test_method(self):
            raise ValueError("test error")
    
    obj = TestClass()
    with pytest.raises(ValueError):
        obj.test_method()
    assert obj.cleaned_up

def test_error_handler_no_error():
    """Test error handler with no error."""
    cleanup_called = False
    
    def cleanup():
        nonlocal cleanup_called
        cleanup_called = True
    
    with ErrorHandler(cleanup_func=cleanup):
        pass
    
    assert not cleanup_called

def test_error_handler_with_error():
    """Test error handler with error."""
    cleanup_called = False
    
    def cleanup():
        nonlocal cleanup_called
        cleanup_called = True
    
    with pytest.raises(ValueError):
        with ErrorHandler(cleanup_func=cleanup):
            raise ValueError("test error")
    
    assert cleanup_called

def test_error_handler_cleanup_error():
    """Test error handler with cleanup error."""
    def cleanup():
        raise RuntimeError("cleanup error")
    
    with pytest.raises(ValueError):
        with ErrorHandler(cleanup_func=cleanup):
            raise ValueError("test error")

def test_safe_api_call_success():
    """Test safe API call with successful response."""
    @safe_api_call
    def api_call():
        return "success"
    
    result = api_call()
    assert result == "success"

def test_safe_api_call_failure():
    """Test safe API call with failed response."""
    @safe_api_call
    def api_call():
        raise requests.exceptions.RequestException("API error")
    
    with pytest.raises(DataFetchError):
        api_call()

def test_safe_api_call_retry():
    """Test safe API call with retry."""
    mock_func = MagicMock(side_effect=[
        requests.exceptions.RequestException("error"),
        "success"
    ])
    
    @safe_api_call
    def api_call():
        return mock_func()
    
    result = api_call()
    assert result == "success"
    assert mock_func.call_count == 2 