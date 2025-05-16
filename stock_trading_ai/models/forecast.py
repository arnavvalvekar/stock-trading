#!/usr/bin/env python3
"""CNN-based time series forecasting model."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Any, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.serialization

# Allow loading of pickled scaler from checkpoint (trusted sources only)
import numpy
torch.serialization.add_safe_globals([MinMaxScaler, numpy._core.multiarray._reconstruct])

logger = logging.getLogger(__name__)

class DummyScaler:
    """Fallback scaler that performs no scaling."""
    def transform(self, X):
        return X

def load_forecast_model(model_path: str, input_size: int = 1, seq_len: int = 60, forecast_horizon: int = 1) -> Tuple[torch.nn.Module, Any]:
    """Load the forecast model and scaler.
    
    Args:
        model_path: Path to the model file
        input_size: Number of input features
        seq_len: Sequence length for prediction
        forecast_horizon: Number of days to forecast
        
    Returns:
        Tuple of (model, scaler)
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            model = checkpoint["model"]
            scaler = checkpoint["scaler"]
        else:
            raise ValueError("Checkpoint is not a dictionary.")

        model.eval()
        return model, scaler

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

        model = torch.nn.Sequential(
            torch.nn.Linear(seq_len, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, forecast_horizon)
        )
        scaler = DummyScaler()
        return model, scaler

def predict_stock(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """Make predictions using the model.
    
    Args:
        model: Trained model
        X: Input features
        
    Returns:
        Model predictions
    """
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).unsqueeze(0)  # shape: (1, 60, 1) or (1, 60)
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
        logger.info(f"Processing data for CNN model. Shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")

        model_path = Path("models/forecast_model.pt")
        if not model_path.exists():
            logger.warning("Model file not found, using default signal")
            return {
                "signal": "hold",
                "confidence": 0.5,
                "prediction": 0.0
            }

        if 'Close' not in data.columns:
            logger.error(f"Missing Close price in data. Available columns: {data.columns.tolist()}")
            return {
                "signal": "hold",
                "confidence": 0.5,
                "prediction": 0.0
            }

        X = data['Close'].values[-60:]
        if len(X) < 60:
            logger.error(f"Insufficient data for prediction. Got {len(X)} days, need 60")
            return {
                "signal": "hold",
                "confidence": 0.5,
                "prediction": 0.0
            }

        logger.info(f"Input data shape for model: {X.shape}")

        model, scaler = load_forecast_model(
            model_path=str(model_path),
            input_size=1,
            seq_len=60,
            forecast_horizon=1
        )

        X_scaled = scaler.transform(X.reshape(-1, 1)).flatten()

        pred = predict_stock(model, X_scaled)
        logger.info(f"Model prediction: {pred}")

        current_price = data['Close'].iloc[-1]
        logger.info(f"Current price: {current_price}")

        price_change = (pred[0] - current_price) / current_price
        logger.info(f"Predicted price change: {price_change:.2%}")

        if price_change > 0.02:
            signal = "buy"
            confidence = min(abs(price_change), 1.0)
        elif price_change < -0.02:
            signal = "sell"
            confidence = min(abs(price_change), 1.0)
        else:
            signal = "hold"
            confidence = 0.5

        logger.info(f"Generated signal: {signal} with confidence {confidence:.2f}")

        return {
            "signal": signal,
            "confidence": confidence,
            "prediction": price_change
        }

    except Exception as e:
        logger.error(f"Error in CNN forecast: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "prediction": 0.0
        }
