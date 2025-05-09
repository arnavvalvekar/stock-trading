"""Macro economic analysis model."""

import os
import requests
from typing import Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def fetch_macro_indicators() -> Dict[str, float]:
    """Fetch macro economic indicators.
    
    Returns:
        Dictionary of indicator values
    """
    try:
        # Use Alpha Vantage API
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.warning("Alpha Vantage API key not found")
            return {}
            
        indicators = {}
        
        # Get GDP data
        url = f"https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        if "data" in data:
            gdp = float(data["data"][0]["value"])
            indicators["gdp"] = gdp
            
        # Get inflation data
        url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        if "data" in data:
            inflation = float(data["data"][0]["value"])
            indicators["inflation"] = inflation
            
        # Get interest rate data
        url = f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=monthly&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        if "data" in data:
            interest_rate = float(data["data"][0]["value"])
            indicators["interest_rate"] = interest_rate
            
        return indicators
        
    except Exception as e:
        logger.error(f"Error fetching macro indicators: {str(e)}")
        return {}

def analyze_macro_conditions(indicators: Dict[str, float]) -> Dict[str, Any]:
    """Analyze macro economic conditions.
    
    Args:
        indicators: Dictionary of indicator values
        
    Returns:
        Dictionary with signal and confidence
    """
    if not indicators:
        return {
            "signal": "hold",
            "confidence": 0.5,
            "indicators": {}
        }
        
    try:
        score = 0
        weights = {
            "gdp": 0.4,
            "inflation": 0.3,
            "interest_rate": 0.3
        }
        
        # Analyze GDP
        if "gdp" in indicators:
            gdp = indicators["gdp"]
            if gdp > 2.0:  # Strong growth
                score += weights["gdp"]
            elif gdp < 0:  # Recession
                score -= weights["gdp"]
                
        # Analyze inflation
        if "inflation" in indicators:
            inflation = indicators["inflation"]
            if 1.5 <= inflation <= 2.5:  # Healthy range
                score += weights["inflation"]
            elif inflation > 3.0:  # Too high
                score -= weights["inflation"]
                
        # Analyze interest rates
        if "interest_rate" in indicators:
            rate = indicators["interest_rate"]
            if 2.0 <= rate <= 4.0:  # Normal range
                score += weights["interest_rate"]
            elif rate > 5.0:  # Too high
                score -= weights["interest_rate"]
                
        # Generate signal
        if score > 0.2:
            signal = "buy"
            confidence = score
        elif score < -0.2:
            signal = "sell"
            confidence = abs(score)
        else:
            signal = "hold"
            confidence = 0.5
            
        return {
            "signal": signal,
            "confidence": confidence,
            "indicators": indicators
        }
        
    except Exception as e:
        logger.error(f"Error analyzing macro conditions: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "indicators": {}
        }

def get_macro_signal() -> Dict[str, Any]:
    """Get macro economic signal.
    
    Returns:
        Dictionary with signal and confidence
    """
    try:
        indicators = fetch_macro_indicators()
        return analyze_macro_conditions(indicators)
    except Exception as e:
        logger.error(f"Error in macro analysis: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "indicators": {}
        } 