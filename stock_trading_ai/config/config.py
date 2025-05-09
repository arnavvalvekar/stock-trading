"""Configuration management for stock trading AI."""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dotenv import load_dotenv

from ..utils.error_handling import ConfigurationError

class Config:
    """Configuration management class."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with default values.
        
        Args:
            config_file: Optional path to a JSON configuration file
        """
        self._config = {
            "api": {
                "alpha_vantage": {
                    "api_key": None
                }
            },
            "models": {
                "save_dir": "models/saved",
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 100
            },
            "data": {
                "dir": "data",  # Base data directory
                "raw_dir": "data/raw",
                "processed_dir": "data/processed"
            },
            "logging": {
                "log_dir": "logs",
                "log_level": "INFO"
            }
        }

        # Load environment variables
        load_dotenv()
        self._load_from_env()

        # Load from file if provided
        if config_file:
            self._load_from_file(config_file)

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            "ALPHA_VANTAGE_API_KEY": "api.alpha_vantage.api_key",
            "MODEL_SAVE_DIR": "models.save_dir",
            "MODEL_BATCH_SIZE": "models.batch_size",
            "MODEL_LEARNING_RATE": "models.learning_rate",
            "MODEL_EPOCHS": "models.epochs",
            "DATA_DIR": "data.dir",
            "LOG_LEVEL": "logging.log_level"
        }

        for env_var, config_key in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                self.set(config_key, value)

    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from a JSON file.
        
        Args:
            config_file: Path to the JSON configuration file
        
        Raises:
            ConfigurationError: If the file cannot be read or parsed
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            self.update(file_config)
        except (IOError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to load config file: {str(e)}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Dot-notation key (e.g., "models.batch_size")
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Dot-notation key (e.g., "models.batch_size")
            value: Value to set
        
        Raises:
            ConfigurationError: If the value is invalid
        """
        # Validate value before setting
        self.validate(key, value)
        
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates, can use dot notation in keys
        """
        def _update_dict(target: Dict[str, Any], source: Dict[str, Any], prefix: str = "") -> None:
            for key, value in source.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    _update_dict(target, value, full_key)
                else:
                    self.set(full_key, value)

        _update_dict(self._config, updates)

    def merge(self, other: 'Config') -> None:
        """Merge another configuration into this one.
        
        Args:
            other: Another Config instance to merge from
        """
        self.update(other._config)

    def save(self, config_file: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            config_file: Path to save the configuration to
        
        Raises:
            ConfigurationError: If the file cannot be written
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self._config, f, indent=4)
        except IOError as e:
            raise ConfigurationError(f"Failed to save config file: {str(e)}")

    def validate(self, key: str, value: Any) -> None:
        """Validate a configuration value.
        
        Args:
            key: Configuration key
            value: Value to validate
        
        Raises:
            ConfigurationError: If the value is invalid
        """
        if key == "models.batch_size":
            if not isinstance(value, (int, str)) or (isinstance(value, str) and not value.isdigit()):
                raise ConfigurationError("Batch size must be a positive integer")
            if isinstance(value, str):
                value = int(value)
            if value <= 0:
                raise ConfigurationError("Batch size must be a positive integer")
        elif key == "models.learning_rate":
            if not isinstance(value, (int, float, str)):
                raise ConfigurationError("Learning rate must be a positive number")
            try:
                float_value = float(value)
                if float_value <= 0:
                    raise ConfigurationError("Learning rate must be a positive number")
            except ValueError:
                raise ConfigurationError("Learning rate must be a valid number")
        elif key == "models.epochs":
            if not isinstance(value, (int, str)) or (isinstance(value, str) and not value.isdigit()):
                raise ConfigurationError("Epochs must be a positive integer")
            if isinstance(value, str):
                value = int(value)
            if value <= 0:
                raise ConfigurationError("Epochs must be a positive integer") 