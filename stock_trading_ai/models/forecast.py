"""CNN-based time series forecasting model."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, seq_len, forecast_horizon, num_filters=32, kernel_size=3, hidden_size=64, num_layers=1):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        x = x.permute(0, 2, 1)  # -> [batch_size, input_size, seq_len]
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # -> [batch_size, seq_len', num_filters]
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

def load_forecast_model(model_path: str, input_size: int, seq_len: int, forecast_horizon: int) -> CNNLSTMModel:
    """Load a trained forecast model.
    
    Args:
        model_path: Path to the saved model
        input_size: Number of input features
        seq_len: Sequence length
        forecast_horizon: Number of steps to forecast
        
    Returns:
        Loaded model
    """
    model = CNNLSTMModel(input_size=input_size, seq_len=seq_len, forecast_horizon=forecast_horizon)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_stock(model: CNNLSTMModel, X: np.ndarray) -> np.ndarray:
    """Make predictions using the model.
    
    Args:
        model: Trained model
        X: Input features
        
    Returns:
        Model predictions
    """
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        pred = model(X_tensor)
        return pred.numpy().flatten()

def get_cnn_signal(data: pd.DataFrame) -> Dict[str, Any]:
    """Get trading signal from CNN model.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Dictionary with signal and confidence
    """
    try:
        # Load the model
        model_path = Path("models/saved/cnn_forecast.pth")
        if not model_path.exists():
            logger.warning("Model file not found, using default signal")
            return {
                "signal": "hold",
                "confidence": 0.5,
                "prediction": 0
            }
            
        model = load_forecast_model(str(model_path), 
                                  input_size=5,  # OHLCV
                                  seq_len=60,    # 60-day window
                                  forecast_horizon=5)
        
        # Prepare input data
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = data[features].values[-60:]  # Last 60 days
        
        # Make prediction
        pred = predict_stock(model, X)
        
        # Convert prediction to signal
        if pred[0] > 0.02:  # 2% threshold
            signal = "buy"
            confidence = pred[0]
        elif pred[0] < -0.02:
            signal = "sell"
            confidence = abs(pred[0])
        else:
            signal = "hold"
            confidence = 0.5
            
        return {
            "signal": signal,
            "confidence": confidence,
            "prediction": pred[0]
        }
    except Exception as e:
        logger.error(f"Error in CNN forecast: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "prediction": 0
        } 