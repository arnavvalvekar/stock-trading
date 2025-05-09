# value evaluation model
from scripts.fetch_fundamentals import get_fundamentals
import math

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def log_transform(val):
    return math.log(val + 1)

def buffet_score(fundamentals):
    weights = {
        "pe_ratio": -0.2,           # Lower P/E is better
        "pb_ratio": -0.1,           # Lower P/B is better
        "roe": 0.3,                 # High ROE is great
        "debt_equity": -0.2,        # Lower D/E is safer
        "current_ratio": 0.1,       # > 1 is ideal
        "revenue_growth": 0.2,      # Strong growth preferred
        "fcf_margin": 0.2           # Good free cash flow margin = healthy business
    }

    score = 0
    active_weight_sum = 0

    for metric, weight in weights.items():
        val = fundamentals.get(metric)

        if val is None or isinstance(val, str):
            print(f"Skipping {metric} due to missing or invalid value")
            continue

        try:
            if metric == "roe":
                val = clamp(val, 0, 25)
                val = log_transform(val)
            elif metric in ["pe_ratio", "pb_ratio"]:
                val = clamp(val, 0, 100)
                val = 1 / (log_transform(val) + 1e-5)  # lower is better
            elif metric == "debt_equity":
                val = clamp(val, 0, 5)
                val = 1 / (val + 1e-5)
            elif metric == "current_ratio":
                val = clamp(val, 0, 5)
            elif metric in ["revenue_growth", "fcf_margin"]:
                val = clamp(val, -1, 1)

            weighted_contribution = weight * val
            score += weighted_contribution
            active_weight_sum += abs(weight)

            print(f"{metric} | val={val:.4f}, weight={weight}, contribution={weighted_contribution:.4f}")

        except Exception as e:
            print(f"Error processing {metric}: {e}")

    if active_weight_sum == 0:
        normalized_score = 0
    else:
        normalized_score = max(0, min(1, score / active_weight_sum))

    if normalized_score > 0.7:
        label = "Undervalued"
    elif normalized_score > 0.4:
        label = "Fairly Valued"
    else:
        label = "Overvalued"

    return {
        "value_score": round(normalized_score, 3),
        "valuation_label": label
    }

def get_value_signal(ticker: str):
    try:
        fundamentals = get_fundamentals(ticker)
        result = buffet_score(fundamentals)

        label = result['valuation_label']
        score = result['value_score']

        if label == "Undervalued":
            signal = "buy"
        elif label == "Overvalued":
            signal = "sell"
        else:
            signal = "hold"

        return {
            "model_name": "value",
            "signal": signal,
            "confidence": round(score, 3)
        }

    except Exception as e:
        print(f"[Value Model Error] Failed to generate signal for {ticker}: {e}")
        return {
            "model_name": "value",
            "signal": "hold",
            "confidence": 0.0
        }

if __name__ == "__main__":
    fundamentals = get_fundamentals("AAPL")
    print("Raw fundamentals:", fundamentals)
    print("Buffett valuation output:", buffet_score(fundamentals))
