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
    try:
        # Calculate technical indicators
        df = data.copy()
        
        # Moving averages
        df['50_MA'] = df['Close'].rolling(window=50).mean()
        df['200_MA'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Generate signals
        signals = []
        
        # Moving average crossover
        if latest['50_MA'] > latest['200_MA']:
            signals.append(("buy", 0.3))
        else:
            signals.append(("sell", 0.3))
            
        # RSI
        if latest['RSI'] < 30:
            signals.append(("buy", 0.2))
        elif latest['RSI'] > 70:
            signals.append(("sell", 0.2))
        else:
            signals.append(("hold", 0.2))
            
        # MACD
        if latest['MACD'] > latest['Signal']:
            signals.append(("buy", 0.2))
        else:
            signals.append(("sell", 0.2))
            
        # Combine signals
        buy_score = sum(weight for signal, weight in signals if signal == "buy")
        sell_score = sum(weight for signal, weight in signals if signal == "sell")
        
        if buy_score > sell_score:
            signal = "buy"
            confidence = buy_score
        elif sell_score > buy_score:
            signal = "sell"
            confidence = sell_score
        else:
            signal = "hold"
            confidence = 0.5
            
        return {
            "signal": signal,
            "confidence": confidence,
            "indicators": {
                "rsi": latest['RSI'],
                "macd": latest['MACD'],
                "signal_line": latest['Signal']
            }
        }
    except Exception as e:
        logger.error(f"Error in technical analysis: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "indicators": {}
        } 