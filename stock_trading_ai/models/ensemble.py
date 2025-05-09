"""Ensemble model for combining multiple trading signals."""

from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def weighted_vote(signals: Dict[str, Dict[str, Any]]) -> Tuple[str, float]:
    """Combine multiple trading signals using weighted voting.
    
    Args:
        signals: Dictionary of model signals
        
    Returns:
        Tuple of (final signal, confidence)
    """
    try:
        # Define model weights
        weights = {
            "CNN Forecast": 0.25,
            "Value Analysis": 0.2,
            "Sentiment": 0.15,
            "Technical": 0.15,
            "Insider": 0.15,
            "Macro": 0.1
        }
        
        # Convert signals to numerical values
        signal_map = {"buy": 1, "hold": 0, "sell": -1}
        weighted_sum = 0
        total_weight = 0
        
        for model_name, signal_data in signals.items():
            weight = weights.get(model_name, 0)
            signal_value = signal_map.get(signal_data["signal"], 0)
            confidence = signal_data.get("confidence", 0.5)
            
            weighted_sum += signal_value * weight * confidence
            total_weight += weight
            
        if total_weight == 0:
            return "hold", 0.5
            
        # Calculate final score
        final_score = weighted_sum / total_weight
        
        # Convert score to signal
        if final_score > 0.2:
            return "buy", abs(final_score)
        elif final_score < -0.2:
            return "sell", abs(final_score)
        else:
            return "hold", 0.5
            
    except Exception as e:
        logger.error(f"Error in ensemble voting: {str(e)}")
        return "hold", 0.5 