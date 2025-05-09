import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load Finnhub API Key from environment variable
API_KEY = os.getenv("FINNHUB_API_KEY")

# --------------------- Data Collection ---------------------

def get_insider_data(ticker):
    url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={ticker}&token={API_KEY}"
    r = requests.get(url)
    try:
        data = r.json()
        print(f"\n📦 Raw insider response for {ticker} (truncated):\n", str(data)[:500])
        return pd.DataFrame(data.get("data", []))
    except Exception as e:
        print(f"❌ Failed to fetch insider data for {ticker}: {e}")
        print("Response content:", r.text)
        return pd.DataFrame()

def get_price_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = None

    if "Close" not in df.columns:
        print(f"❌ 'Close' column not found in price data for {ticker}. Columns: {df.columns.tolist()}")
        return pd.DataFrame()
    return df[["Close"]]

# --------------------- Feature Engineering ---------------------

def process_insider_data(df):
    df = df[df["transactionPrice"] > 0].copy()
    df["transactionDate"] = pd.to_datetime(df["transactionDate"])
    df["shares"] = df["change"].astype(float)
    df["buy"] = df["transactionCode"].isin(["P", "A", "M"])  # Buy-type codes

    grouped = df.groupby("transactionDate").agg(
        total_shares=("shares", "sum"),
        buy_ratio=("buy", "mean")
    )
    return grouped

def generate_labels(price_df, horizon=5, threshold=0.02):
    price_df = price_df.copy()
    price_df["future_close"] = price_df["Close"].shift(-horizon)
    price_df["pct_change"] = (price_df["future_close"] - price_df["Close"]) / price_df["Close"]
    price_df["label"] = (price_df["pct_change"] > threshold).astype(int)
    return price_df[["label"]]

# --------------------- Dataset Creation ---------------------

def build_dataset(ticker):
    end = datetime.today()
    start = end - timedelta(days=365)

    insider_df = get_insider_data(ticker)
    price_df = get_price_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    print(f"\n🔎 Data check for {ticker}:")
    print("Insider Data:", insider_df.shape)
    print("Price Data:", price_df.shape)

    if insider_df.empty or price_df.empty:
        print("⚠️ Missing data for", ticker)
        return None, None, None

    insider_features = process_insider_data(insider_df)

    # Align indexes for merging
    if isinstance(price_df.index, pd.MultiIndex):
        price_df.index = price_df.index.get_level_values(0)

    df = price_df.merge(insider_features, left_index=True, right_index=True, how="left")
    df = df.fillna(method="ffill").dropna()

    labels = generate_labels(df)
    df = df.merge(labels, left_index=True, right_index=True, how="inner")
    df = df.drop(columns=["future_close", "pct_change"], errors="ignore")

    X = df[["total_shares", "buy_ratio"]]
    y = df["label"]

    return X, y, df.index

# --------------------- Model Training & Inference ---------------------

_model = None

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def get_insider_signal(ticker: str) -> int:
    global _model

    X, y, index = build_dataset(ticker)
    if X is None or X.empty:
        return 0  # fallback = bearish

    if _model is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        _model = train_model(X_train, y_train)
        y_pred = _model.predict(X_test)
        print("📊 Insider Model Performance:")
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    latest_features = X.iloc[[-1]]
    signal = _model.predict(latest_features)[0]
    return int(signal)

def get_insider_signal_wrapped(ticker: str):
    try:
        raw = get_insider_signal(ticker)
        signal = "buy" if raw == 1 else "sell"
        confidence = 0.7 if signal == "buy" else 0.6
        
        return {
            "model_name": "insider",
            "signal": signal,
            "confidence": confidence
        }

    except Exception as e:
        print(f"[Insider Model Error] {e}")
        return {"model_name": "insider", "signal": "hold", "confidence": 0.0}


if __name__ == "__main__":
    ticker = "AAPL"
    signal = get_insider_signal(ticker)
    print(f"📈 Insider signal for {ticker}: {'Bullish' if signal == 1 else 'Bearish'}")
