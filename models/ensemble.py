# combines all models into one

from models.forecast import get_cnn_signal
from models.value_model import get_value_signal
from models.sentiment import get_sentiment_signal
from models.insider_model import get_insider_signal_wrapped as get_insider_signal
from models.technical_patterns import get_technical_signal
from models.macro_model import get_macro_signal

import pandas as pd
import os
from datetime import datetime

WEIGHTS = {
    "cnn": 0.25,
    "value": 0.2,
    "sentiment": 0.15,
    "technical": 0.15,
    "insider": 0.15,
    "macro": 0.1
}

def weighted_vote(signals):
    score_map = {"buy": 1, "hold": 0, "sell": -1}
    total_score = 0
    total_weight = 0

    for s in signals:
        weight = WEIGHTS.get(s["model_name"], 0)
        score = score_map.get(s["signal"], 0)
        total_score += score * weight
        total_weight += weight

    avg_score = total_score / total_weight if total_weight else 0

    if avg_score > 0.25:
        return "buy", avg_score
    elif avg_score < -0.25:
        return "sell", avg_score
    else:
        return "hold", avg_score

def log_decision(ticker, signals, final_signal, confidence, log_path="logs/ensemble_log.csv"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": timestamp,
        "ticker": ticker,
        "final_signal": final_signal,
        "ensemble_confidence": round(confidence, 3),
    }

    for s in signals:
        row[f"{s['model_name']}_signal"] = s["signal"]
        row[f"{s['model_name']}_conf"] = s["confidence"]

    df = pd.DataFrame([row])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', index=False, header=False)
    else:
        df.to_csv(log_path, index=False)

def run_ensemble(ticker: str):
    signals = [
        get_cnn_signal(ticker),
        get_value_signal(ticker),
        get_sentiment_signal(ticker),
        get_technical_signal(ticker),
        get_insider_signal(ticker),
        get_macro_signal(ticker)
    ]

    final_signal, confidence = weighted_vote(signals)
    log_decision(ticker, signals, final_signal, confidence)

    return final_signal, confidence, signals

if __name__ == "__main__":
    ticker = "AAPL"
    decision, conf, model_outputs = run_ensemble(ticker)

    print(f"\nFinal decision for {ticker}: {decision.upper()} (Confidence: {conf:.3f})\n")
    print("Model Breakdown:")
    for m in model_outputs:
        print(f" - {m['model_name']}: {m['signal']} (conf: {m['confidence']})")
