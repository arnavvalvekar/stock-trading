# pull fundamental information for value model

import os
import finnhub

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

def get_fundamentals(ticker):
    profile = finnhub_client.company_profile2(symbol=ticker)
    metrics = finnhub_client.company_basic_financials(symbol=ticker, metric="all")

    data = metrics.get("metric", {})

    return {
        "ticker": ticker,
        "pe_ratio": data.get("peInclExtraTTM"),
        "pb_ratio": data.get("pbAnnual"),
        "roe": data.get("roeTTM"),
        "debt_equity": data.get("totalDebt/totalEquityAnnual"),
        "current_ratio": data.get("currentRatioAnnual"),
        "revenue_growth": data.get("revenueGrowthTTM"),
        "fcf_margin": data.get("freeCashFlowMarginTTM"),
    }

if __name__ == "__main__":
    print(get_fundamentals("AAPL"))
