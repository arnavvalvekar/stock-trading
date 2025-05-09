"""Insider trading analysis model."""

import os
import requests
from typing import Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def fetch_insider_trades(symbol: str, days_back: int = 30) -> list:
    """Fetch insider trading data for a stock.
    
    Args:
        symbol: Stock symbol
        days_back: Number of days to look back
        
    Returns:
        List of insider trades
    """
    try:
        # Use Alpha Vantage API
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.warning("Alpha Vantage API key not found")
            return []
            
        url = f"https://www.alphavantage.co/query?function=INSIDER_TRADING&symbol={symbol}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "insiderTrades" not in data:
            logger.warning(f"No insider trading data found for {symbol}")
            return []
            
        # Filter recent trades
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        trades = []
        for trade in data["insiderTrades"]:
            trade_date = datetime.strptime(trade["transactionDate"], "%Y-%m-%d")
            if start_date <= trade_date <= end_date:
                trades.append(trade)
                
        return trades
        
    except Exception as e:
        logger.error(f"Error fetching insider trades for {symbol}: {str(e)}")
        return []

def analyze_insider_trades(trades: list) -> Dict[str, Any]:
    """Analyze insider trading patterns.
    
    Args:
        trades: List of insider trades
        
    Returns:
        Dictionary with signal and confidence
    """
    if not trades:
        return {
            "signal": "hold",
            "confidence": 0.5,
            "buy_volume": 0,
            "sell_volume": 0
        }
        
    try:
        buy_volume = 0
        sell_volume = 0
        
        for trade in trades:
            volume = float(trade["transactionShares"])
            if trade["transactionType"] == "P":  # Purchase
                buy_volume += volume
            elif trade["transactionType"] == "S":  # Sale
                sell_volume += volume
                
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return {
                "signal": "hold",
                "confidence": 0.5,
                "buy_volume": 0,
                "sell_volume": 0
            }
            
        # Calculate buy/sell ratio
        ratio = buy_volume / total_volume
        
        # Generate signal
        if ratio > 0.7:  # Strong buying
            signal = "buy"
            confidence = ratio
        elif ratio < 0.3:  # Strong selling
            signal = "sell"
            confidence = 1 - ratio
        else:
            signal = "hold"
            confidence = 0.5
            
        return {
            "signal": signal,
            "confidence": confidence,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume
        }
        
    except Exception as e:
        logger.error(f"Error analyzing insider trades: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "buy_volume": 0,
            "sell_volume": 0
        }

def get_insider_signal_wrapped(symbol: str) -> Dict[str, Any]:
    """Get insider trading signal for a stock.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with signal and confidence
    """
    try:
        trades = fetch_insider_trades(symbol)
        return analyze_insider_trades(trades)
    except Exception as e:
        logger.error(f"Error in insider analysis for {symbol}: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "buy_volume": 0,
            "sell_volume": 0
        } 