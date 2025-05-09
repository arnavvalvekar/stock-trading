# sentiment analysis model

import os
import requests
from transformers import pipeline
from datetime import datetime, timedelta
from dotenv import load_dotenv
import csv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def fetch_news(symbol, days_back=3, max_articles=50):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date.date()}&to={end_date.date()}&token={FINNHUB_API_KEY}"

    res = requests.get(url)
    news_items = res.json()

    if not isinstance(news_items, list):
        raise ValueError(f"Unexpected API response: {news_items}")

    headlines = [item['headline'] for item in news_items[:max_articles]]
    return headlines

def get_sentiment_score(texts):
    results = sentiment_model(texts)
    scores = []
    for r in results:
        score = r["score"] if r["label"] == "POSITIVE" else -r["score"]
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0

def analyze_stock_sentiment(symbol):
    print(f"Fetching news for: {symbol}")
    try:
        headlines = fetch_news(symbol)
        sentiment = get_sentiment_score(headlines)
        print(f"Sentiment score for {symbol}: {sentiment}")
        return sentiment
    except Exception as e:
        print(f"Error analyzing sentiment for {symbol}: {e}")
        return None

def get_sentiment_signal(ticker: str):
    try:
        sentiment_score = analyze_stock_sentiment(ticker)

        if sentiment_score > 0.2:
            signal = "buy"
        elif sentiment_score < -0.2:
            signal = "sell"
        else:
            signal = "hold"

        return {
            "model_name": "sentiment",
            "signal": signal,
            "confidence": round(abs(sentiment_score), 3)
        }

    except Exception as e:
        print(f"[Sentiment Model Error] Failed to generate signal for {ticker}: {e}")
        return {
            "model_name": "sentiment",
            "signal": "hold",
            "confidence": 0.0
        }


if __name__ == "__main__":
    analyze_stock_sentiment("AAPL")
