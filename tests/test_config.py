import pytest
import os
import json
from pathlib import Path
import tempfile

from stock_trading_ai.config import Config
from stock_trading_ai.utils.error_handling import ConfigurationError

class TestConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        # Check default values
        assert config.get("api.alpha_vantage.api_key") is None
        assert config.get("models.save_dir") == "models/saved"
        assert config.get("data.dir") == "data"
        assert config.get("logging.level") == "INFO"

    def test_set_get_config(self):
        """Test setting and getting configuration values."""
        config = Config()
        
        # Set values
        config.set("test.key1", "value1")
        config.set("test.key2", 42)
        config.set("test.key3", True)
        
        # Get values
        assert config.get("test.key1") == "value1"
        assert config.get("test.key2") == 42
        assert config.get("test.key3") is True

    def test_nested_config(self):
        """Test nested configuration structure."""
        config = Config()
        
        # Set nested values
        config.set("models.cnn.input_size", 10)
        config.set("models.cnn.num_classes", 3)
        config.set("models.cnn.learning_rate", 0.001)
        
        # Get nested values
        assert config.get("models.cnn.input_size") == 10
        assert config.get("models.cnn.num_classes") == 3
        assert config.get("models.cnn.learning_rate") == 0.001

    def test_config_from_env(self, mock_env_vars):
        """Test loading configuration from environment variables."""
        config = Config()
        
        # Check environment variables
        assert config.get("api.alpha_vantage.api_key") == "test_key"
        assert config.get("models.save_dir") == "test_models"
        assert config.get("data.dir") == "test_data"
        assert config.get("logging.level") == "DEBUG"

    def test_config_from_file(self):
        """Test loading configuration from file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "api": {
                    "alpha_vantage": {
                        "api_key": "file_key"
                    }
                },
                "models": {
                    "save_dir": "file_models"
                }
            }, f)
            config_path = f.name
        
        try:
            # Load config from file
            config = Config(config_path)
            
            # Check values
            assert config.get("api.alpha_vantage.api_key") == "file_key"
            assert config.get("models.save_dir") == "file_models"
        finally:
            # Cleanup
            os.unlink(config_path)

    def test_config_save(self):
        """Test saving configuration to file."""
        config = Config()
        
        # Set some values
        config.set("test.key1", "value1")
        config.set("test.key2", 42)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config_path = f.name
        
        try:
            # Save config
            config.save(config_path)
            
            # Load config from file
            loaded_config = Config(config_path)
            
            # Check values
            assert loaded_config.get("test.key1") == "value1"
            assert loaded_config.get("test.key2") == 42
        finally:
            # Cleanup
            os.unlink(config_path)

    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        with pytest.raises(ConfigurationError):
            Config("nonexistent_config.json")

    def test_invalid_config_value(self):
        """Test handling of invalid configuration values."""
        config = Config()
        
        with pytest.raises(ConfigurationError):
            config.set("", "value")
        
        with pytest.raises(ConfigurationError):
            config.set("key", None)

    def test_config_update(self):
        """Test updating configuration values."""
        config = Config()
        
        # Set initial values
        config.set("test.key1", "value1")
        config.set("test.key2", 42)
        
        # Update values
        config.update({
            "test.key1": "new_value1",
            "test.key2": 100,
            "test.key3": "value3"
        })
        
        # Check updated values
        assert config.get("test.key1") == "new_value1"
        assert config.get("test.key2") == 100
        assert config.get("test.key3") == "value3"

    def test_config_merge(self):
        """Test merging configurations."""
        config1 = Config()
        config2 = Config()
        
        # Set values in first config
        config1.set("test.key1", "value1")
        config1.set("test.key2", 42)
        
        # Set values in second config
        config2.set("test.key2", 100)
        config2.set("test.key3", "value3")
        
        # Merge configs
        config1.merge(config2)
        
        # Check merged values
        assert config1.get("test.key1") == "value1"
        assert config1.get("test.key2") == 100
        assert config1.get("test.key3") == "value3" 