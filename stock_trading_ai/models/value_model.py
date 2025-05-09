"""Value analysis model for stock trading."""

import yfinance as yf
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_value_signal(symbol: str) -> Dict[str, Any]:
    """Get trading signal from value analysis.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with signal and confidence
    """
    try:
        # Get stock info
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Calculate value metrics
        pe_ratio = info.get('trailingPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        dividend_yield = info.get('dividendYield', 0)
        
        # Simple value scoring
        score = 0
        if pe_ratio > 0 and pe_ratio < 15:
            score += 1
        if pb_ratio > 0 and pb_ratio < 2:
            score += 1
        if dividend_yield > 0.03:  # 3% threshold
            score += 1
            
        # Convert score to signal
        if score >= 2:
            signal = "buy"
            confidence = score / 3
        elif score == 1:
            signal = "hold"
            confidence = 0.5
        else:
            signal = "sell"
            confidence = 1 - (score / 3)
            
        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "dividend_yield": dividend_yield
            }
        }
    except Exception as e:
        logger.error(f"Error in value analysis: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "metrics": {}
        } 