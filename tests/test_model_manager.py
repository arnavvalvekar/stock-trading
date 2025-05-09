"""Tests for the ModelManager class."""

import os
import pytest
import torch
import tempfile
from pathlib import Path

from stock_trading_ai.config import Config
from stock_trading_ai.models import ModelManager
from stock_trading_ai.utils.error_handling import ModelError

@pytest.fixture
def config():
    """Create a test configuration."""
    config = Config()
    config.set("model.save_dir", str(Path(tempfile.mkdtemp()) / "models"))
    return config

@pytest.fixture
def model_manager(config):
    """Create a test model manager."""
    return ModelManager(config)

@pytest.fixture
def test_data():
    """Create test data."""
    return torch.randn(10, 10), torch.randint(0, 3, (10,))

def test_model_creation(model_manager):
    """Test model creation."""
    model_manager.create_model("cnn")
    assert model_manager.model is not None

def test_model_saving_loading(model_manager):
    """Test model saving and loading."""
    # Create and save model
    model_manager.create_model("cnn")
    model_manager.save_model("test_model.pt")

    # Load model
    model_manager.load_model("test_model.pt", "cnn")
    assert model_manager.model is not None

def test_model_evaluation(model_manager, test_data):
    """Test model evaluation."""
    data, labels = test_data
    model_manager.create_model("cnn")
    metrics = model_manager.evaluate_model(data, labels)
    
    assert "loss" in metrics
    assert "accuracy" in metrics

def test_model_history(model_manager):
    """Test model history tracking."""
    model_manager.create_model("cnn")
    
    # Update history
    metrics = {"loss": 0.5, "accuracy": 0.8}
    model_manager.update_history(metrics)
    
    history = model_manager.get_history()
    assert "loss" in history
    assert "accuracy" in history
    assert history["loss"] == [0.5]
    assert history["accuracy"] == [0.8]

def test_invalid_model_name(model_manager):
    """Test creating model with invalid name."""
    with pytest.raises(ModelError):
        model_manager.create_model("invalid_model")

def test_model_not_found(model_manager):
    """Test loading non-existent model."""
    with pytest.raises(ModelError):
        model_manager.load_model("nonexistent.pt", "cnn")

@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_batch_processing(model_manager, batch_size):
    """Test batch processing with different sizes."""
    model_manager.create_model("cnn")
    data = torch.randn(batch_size, 10)
    output = model_manager.process_batch(data, batch_size)
    assert output.shape[0] == batch_size

def test_model_cleanup(model_manager):
    """Test model cleanup."""
    model_manager.create_model("cnn")
    assert model_manager.model is not None
    
    model_manager.cleanup()
    assert model_manager.model is None
    assert model_manager.model_history == {} 