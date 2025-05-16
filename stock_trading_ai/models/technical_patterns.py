"""Technical analysis model for stock trading."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_technical_signal(data: pd.DataFrame) -> Dict[str, Any]:
    """Get trading signal from technical analysis.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Dictionary with signal and confidence
    """
    data = data.apply(pd.to_numeric, errors='coerce')
    try:
        if data.empty:
            return {
                "signal": "hold",
                "confidence": 0.5,
                "indicators": {}
            }
            
        # Calculate technical indicators
        indicators = {}
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['RSI'] = rsi.iloc[-1]
        
        # Moving Averages
        data['50_MA'] = data['Close'].rolling(window=50).mean()
        data['200_MA'] = data['Close'].rolling(window=200).mean()
        indicators['50_MA'] = data['50_MA'].iloc[-1]
        indicators['200_MA'] = data['200_MA'].iloc[-1]
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        indicators['MACD'] = macd.iloc[-1]
        indicators['MACD_Signal'] = macd_signal.iloc[-1]
        
        # Generate signal based on indicators
        trading_signal = "hold"
        confidence = 0.5
        
        # RSI signals
        if rsi.iloc[-1] > 70:
            trading_signal = "sell"
            confidence = min((rsi.iloc[-1] - 70) / 30, 1.0)
        elif rsi.iloc[-1] < 30:
            trading_signal = "buy"
            confidence = min((30 - rsi.iloc[-1]) / 30, 1.0)
            
        # Moving Average signals
        if data['50_MA'].iloc[-1] > data['200_MA'].iloc[-1] and data['50_MA'].iloc[-2] <= data['200_MA'].iloc[-2]:
            trading_signal = "buy"
            confidence = max(confidence, 0.7)
        elif data['50_MA'].iloc[-1] < data['200_MA'].iloc[-1] and data['50_MA'].iloc[-2] >= data['200_MA'].iloc[-2]:
            trading_signal = "sell"
            confidence = max(confidence, 0.7)
            
        # MACD signals
        if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
            trading_signal = "buy"
            confidence = max(confidence, 0.6)
        elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
            trading_signal = "sell"
            confidence = max(confidence, 0.6)
            
        return {
            "signal": trading_signal,
            "confidence": confidence,
            "indicators": indicators
        }
        
    except Exception as e:
        logger.error(f"Error in technical analysis: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "indicators": {}
        } 