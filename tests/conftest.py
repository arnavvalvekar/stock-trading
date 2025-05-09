import pytest
import os
import tempfile
from pathlib import Path
import shutil

from stock_trading_ai.config import Config

@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration with temporary directories."""
    config = Config()
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp(prefix="stock_trading_ai_test_"))
    data_dir = temp_dir / "data"
    model_dir = temp_dir / "models"
    log_dir = temp_dir / "logs"
    
    # Create directories
    data_dir.mkdir()
    model_dir.mkdir()
    log_dir.mkdir()
    
    # Update configuration
    config.set("data.dir", str(data_dir))
    config.set("models.save_dir", str(model_dir))
    config.set("logging.dir", str(log_dir))
    
    # Set test API key
    config.set("api.alpha_vantage.api_key", "test_key")
    
    yield config
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp(prefix="stock_trading_ai_data_"))
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def temp_model_dir():
    """Create a temporary directory for test models."""
    temp_dir = Path(tempfile.mkdtemp(prefix="stock_trading_ai_models_"))
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    monkeypatch.setenv("MODEL_SAVE_DIR", "test_models")
    monkeypatch.setenv("DATA_DIR", "test_data")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

@pytest.fixture(scope="function")
def sample_stock_data():
    """Create sample stock data for testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = {
        "Open": np.random.randn(len(dates)).cumsum() + 100,
        "High": np.random.randn(len(dates)).cumsum() + 102,
        "Low": np.random.randn(len(dates)).cumsum() + 98,
        "Close": np.random.randn(len(dates)).cumsum() + 100,
        "Volume": np.random.randint(1000, 10000, len(dates))
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture(scope="function")
def sample_model_config():
    """Create sample model configuration for testing."""
    return {
        "cnn": {
            "input_size": 10,
            "num_classes": 3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        },
        "value": {
            "window_size": 20,
            "threshold": 0.05
        },
        "sentiment": {
            "max_length": 512,
            "batch_size": 16
        }
    } 