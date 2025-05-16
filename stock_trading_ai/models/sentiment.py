"""Sentiment analysis model for stock trading."""

import os
import requests
from typing import Dict, Any
import logging
from transformers import pipeline
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

# Initialize sentiment model
try:
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    logger.warning(f"Failed to load sentiment model: {str(e)}")
    sentiment_model = None

def fetch_news(symbol: str, days_back: int = 3, max_articles: int = 50) -> list:
    """Fetch news articles for a stock symbol.
    
    Args:
        symbol: Stock symbol
        days_back: Number of days to look back
        max_articles: Maximum number of articles to fetch
        
    Returns:
        List of news headlines
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Use Alpha Vantage News API
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            logger.warning("Alpha Vantage API key not found")
            return []
            
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "feed" not in data:
            logger.warning(f"No news data found for {symbol}")
            return []
            
        headlines = [item["title"] for item in data["feed"][:max_articles]]
        return headlines
        
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return []

def get_sentiment_score(texts: list) -> float:
    """Get sentiment score for a list of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Average sentiment score (-1 to 1)
    """
    if not sentiment_model or not texts:
        return 0.0
        
    try:
        results = sentiment_model(texts)
        scores = []
        for r in results:
            score = r["score"] if r["label"] == "POSITIVE" else -r["score"]
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return 0.0

def get_sentiment_signal(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Get trading signal from sentiment analysis.
    
    Args:
        data: DataFrame with price data
        symbol: Stock symbol
        
    Returns:
        Dictionary with signal and confidence
    """
    try:
        # Check if Alpha Vantage API key is available
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            logger.warning("Alpha Vantage API key not found. Using default sentiment values.")
            return {
                "signal": "hold",
                "confidence": 0.5,
                "sentiment": {
                    "overall": 0.0,
                    "news": [],
                    "insider": []
                }
            }
            
        # Get news sentiment
        news_sentiment = get_news_sentiment(symbol, api_key)
        
        # Get insider sentiment
        insider_sentiment = get_insider_sentiment(symbol, api_key)
        
        # Combine sentiments
        overall_sentiment = 0.0
        if news_sentiment:
            overall_sentiment += news_sentiment.get('overall', 0.0)
        if insider_sentiment:
            overall_sentiment += insider_sentiment.get('overall', 0.0)
            
        # Normalize sentiment to [-1, 1]
        overall_sentiment = max(min(overall_sentiment / 2, 1.0), -1.0)
        
        # Generate signal
        if overall_sentiment > 0.2:
            signal = "buy"
            confidence = min((overall_sentiment + 1) / 2, 1.0)
        elif overall_sentiment < -0.2:
            signal = "sell"
            confidence = min((1 - overall_sentiment) / 2, 1.0)
        else:
            signal = "hold"
            confidence = 0.5
            
        return {
            "signal": signal,
            "confidence": confidence,
            "sentiment": {
                "overall": overall_sentiment,
                "news": news_sentiment.get('articles', []) if news_sentiment else [],
                "insider": insider_sentiment.get('transactions', []) if insider_sentiment else []
            }
        }
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "sentiment": {
                "overall": 0.0,
                "news": [],
                "insider": []
            }
        } 