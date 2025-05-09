# model to decide pattern detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
TOLERANCE = 0.01          # Max price difference between tops/bottoms (1%)
MIN_SEPARATION = 5        # Min days between first and second peak/trough
CONFIRMATION_WINDOW = 3   # Days to confirm post-pattern reversal

def detect_double_top(df, tolerance=TOLERANCE, min_separation=MIN_SEPARATION):
    close = df["Close"]
    peaks = (close.shift(1) < close) & (close.shift(-1) < close)
    peak_indices = close[peaks].index.to_list()

    patterns = []
    for i in range(len(peak_indices) - 1):
        first_idx = peak_indices[i]
        second_idx = peak_indices[i + 1]

        if (second_idx - first_idx).days < min_separation:
            continue

        first_price = close[first_idx]
        second_price = close[second_idx]

        if abs(first_price - second_price) / first_price < tolerance:
            mid = df.loc[first_idx:second_idx]["Close"].min()
            if mid < min(first_price, second_price):
                confirmation_start = second_idx + pd.Timedelta(days=1)
                confirmation_end = confirmation_start + pd.Timedelta(days=CONFIRMATION_WINDOW)
                if confirmation_start in df.index and confirmation_end in df.index:
                    post_confirm_price = df.loc[confirmation_end, "Close"]
                    if post_confirm_price < mid:
                        patterns.append((first_idx, second_idx))
    return patterns


def detect_double_bottom(df, tolerance=TOLERANCE, min_separation=MIN_SEPARATION):
    close = df["Close"]
    troughs = (close.shift(1) > close) & (close.shift(-1) > close)
    trough_indices = close[troughs].index.to_list()

    patterns = []
    for i in range(len(trough_indices) - 1):
        first_idx = trough_indices[i]
        second_idx = trough_indices[i + 1]

        if (second_idx - first_idx).days < min_separation:
            continue

        first_price = close[first_idx]
        second_price = close[second_idx]

        if abs(first_price - second_price) / first_price < tolerance:
            mid = df.loc[first_idx:second_idx]["Close"].max()
            if mid > max(first_price, second_price):
                confirmation_start = second_idx + pd.Timedelta(days=1)
                confirmation_end = confirmation_start + pd.Timedelta(days=CONFIRMATION_WINDOW)
                if confirmation_start in df.index and confirmation_end in df.index:
                    post_confirm_price = df.loc[confirmation_end, "Close"]
                    if post_confirm_price > mid:
                        patterns.append((first_idx, second_idx))
    return patterns


def plot_pattern(df, pattern_indices, pattern_type):
    for i, (start, end) in enumerate(pattern_indices):
        plt.figure(figsize=(10, 4))
        df['Close'].loc[start:end].plot(title=f"{pattern_type} from {start} to {end}")
        plt.axvline(start, color='r', linestyle='--')
        plt.axvline(end, color='r', linestyle='--')
        plt.tight_layout()
        plt.show()


def analyze_patterns(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    print("Analyzing:", csv_path)

    dtops = detect_double_top(df)
    print("\nDouble Tops:")
    for start, end in dtops:
        print(f"From {start} to {end}")
    plot_pattern(df, dtops, "Double Top")

    dbottoms = detect_double_bottom(df)
    print("\nDouble Bottoms:")
    for start, end in dbottoms:
        print(f"From {start} to {end}")
    plot_pattern(df, dbottoms, "Double Bottom")

def get_pattern_signals(df):
    double_tops = detect_double_top(df)
    double_bottoms = detect_double_bottom(df)

    signal_data = {
        "double_tops": [{"start": str(start), "end": str(end)} for start, end in double_tops],
        "double_bottoms": [{"start": str(start), "end": str(end)} for start, end in double_bottoms],
        "score": len(double_bottoms) - len(double_tops)  # crude bullish vs bearish sentiment
    }

    return signal_data

def get_technical_signal(ticker: str):
    try:
        path = f"data/price_data/{ticker}_price_data.csv"
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

        signals = get_pattern_signals(df)
        score = signals["score"]

        if score > 0:
            signal = "buy"
        elif score < 0:
            signal = "sell"
        else:
            signal = "hold"

        confidence = min(1.0, abs(score) / 3)  # normalize confidence
        return {
            "model_name": "technical",
            "signal": signal,
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        print(f"[Technical Model Error] {e}")
        return {"model_name": "technical", "signal": "hold", "confidence": 0.0}

if __name__ == "__main__":
    df = pd.read_csv("data/price_data/AAPL_price_data.csv", parse_dates=['Date'], index_col='Date')
    analyze_patterns("data/price_data/AAPL_price_data.csv")

    signals = get_pattern_signals(df)
    print("\nStructured Signals:")
    print(signals)
