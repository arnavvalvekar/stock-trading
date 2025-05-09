"""Model architectures for stock trading AI."""

import torch
import torch.nn as nn
from typing import Dict, Any

class TimeSeriesCNN(nn.Module):
    """CNN architecture for time series forecasting."""
    
    def __init__(self, input_size: int = 10, num_classes: int = 3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate the size of the flattened features
        self.flatten_size = 128 * (input_size // 8)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def create_model_architecture(model_name: str, **kwargs) -> nn.Module:
    """
    Create a model architecture based on the model name.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments for model creation
        
    Returns:
        PyTorch model instance
    """
    model_configs: Dict[str, Dict[str, Any]] = {
        "cnn": {
            "class": TimeSeriesCNN,
            "default_args": {
                "input_size": 10,
                "num_classes": 3
            }
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model type: {model_name}")
    
    config = model_configs[model_name]
    model_class = config["class"]
    default_args = config["default_args"]
    
    # Update default args with provided kwargs
    model_args = {**default_args, **kwargs}
    
    return model_class(**model_args) 