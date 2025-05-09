"""Tests for logging configuration."""

import os
import logging
import pytest
from pathlib import Path
import tempfile
import shutil

from stock_trading_ai.config.logging_config import setup_logging, get_logger

@pytest.fixture
def temp_log_dir():
    """Create a temporary log directory."""
    temp_dir = Path(tempfile.mkdtemp(prefix="stock_trading_ai_logs_"))
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_setup_logging(temp_log_dir):
    """Test basic logging setup."""
    # Set up logging
    setup_logging(
        log_dir=str(temp_log_dir),
        log_level="DEBUG",
        log_file="test.log"
    )
    
    # Check if log file was created
    log_file = temp_log_dir / "test.log"
    assert log_file.exists()
    
    # Check if logger was configured correctly
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2  # Console and file handlers

def test_invalid_log_level(temp_log_dir):
    """Test handling of invalid log level."""
    with pytest.raises(ValueError):
        setup_logging(
            log_dir=str(temp_log_dir),
            log_level="INVALID"
        )

def test_log_rotation(temp_log_dir):
    """Test log file rotation."""
    # Set up logging with small maxBytes to trigger rotation
    setup_logging(
        log_dir=str(temp_log_dir),
        log_level="DEBUG",
        log_file="rotation_test.log",
        max_bytes=100,  # Small size to trigger rotation
        backup_count=3
    )
    
    logger = get_logger("test_logger")
    
    # Write enough logs to trigger rotation
    for i in range(100):
        logger.debug("x" * 10)  # Each log line is at least 10 bytes
    
    # Check if backup files were created
    log_files = list(temp_log_dir.glob("rotation_test.log*"))
    assert len(log_files) > 1

def test_get_logger():
    """Test getting logger instances."""
    logger1 = get_logger("test_logger")
    logger2 = get_logger("test_logger")
    
    # Same name should return same logger instance
    assert logger1 is logger2
    assert logger1.name == "test_logger"

def test_log_output(temp_log_dir, caplog):
    """Test log output format."""
    setup_logging(
        log_dir=str(temp_log_dir),
        log_level="DEBUG"
    )

    logger = get_logger("test_logger")
    test_message = "Test log message"

    # Log messages at different levels
    logger.debug(test_message)
    logger.info(test_message)
    logger.warning(test_message)
    logger.error(test_message)
    logger.critical(test_message)

    # Check if messages were logged
    assert len(caplog.records) == 5
    for record in caplog.records:
        assert record.msg == test_message

def test_log_directory_creation():
    """Test automatic creation of log directory."""
    temp_dir = Path(tempfile.mkdtemp())
    log_dir = temp_dir / "logs"
    
    try:
        # Directory should not exist
        assert not log_dir.exists()
        
        # Setup logging should create directory
        setup_logging(log_dir=str(log_dir))
        
        # Directory should now exist
        assert log_dir.exists()
    finally:
        shutil.rmtree(temp_dir) 