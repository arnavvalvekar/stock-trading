"""Value analysis model for stock trading."""

import os
import finnhub
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

def get_value_signal(symbol: str) -> Dict[str, Any]:
    """Get trading signal from value analysis.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with signal and confidence
    """
    try:
        # Get company profile and financials from Finnhub
        profile = finnhub_client.company_profile2(symbol=symbol)
        metrics = finnhub_client.company_basic_financials(symbol=symbol, metric="all")
        
        # Extract metrics
        pe_ratio = metrics.get('metric', {}).get('peInclExtraTTM', 0)
        pb_ratio = metrics.get('metric', {}).get('pbAnnual', 0)
        dividend_yield = metrics.get('metric', {}).get('dividendYieldIndicatedAnnual', 0)
        
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