# script to preprocess data for an individual stock

import os
import pandas as pd
import finnhub
from datetime import datetime, timedelta

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

# csv creation
ticker = "AAPL"
start_date = "2015-01-01"
end_date = "2025-01-01"

# Convert dates to timestamps
start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

# price data file
stock_data = finnhub_client.stock_candles(
    ticker,
    'D',  # Daily interval
    start_timestamp,
    end_timestamp
)

if stock_data and stock_data.get('s') == 'ok':
    price_df = pd.DataFrame({
        'Open': stock_data['o'],
        'High': stock_data['h'],
        'Low': stock_data['l'],
        'Close': stock_data['c'],
        'Volume': stock_data['v']
    }, index=pd.to_datetime(stock_data['t'], unit='s'))
    
    price_csv_path = os.path.join("data/price_data", f"{ticker}_price_data.csv")
    price_df.to_csv(price_csv_path)

# fundamental data file
profile = finnhub_client.company_profile2(symbol=ticker)
metrics = finnhub_client.company_basic_financials(symbol=ticker, metric="all")

fundamental_data = {
    'ticker': ticker,
    'name': profile.get('name'),
    'market_cap': profile.get('marketCapitalization'),
    'pe_ratio': metrics.get('metric', {}).get('peInclExtraTTM'),
    'pb_ratio': metrics.get('metric', {}).get('pbAnnual'),
    'roe': metrics.get('metric', {}).get('roeTTM'),
    'debt_equity': metrics.get('metric', {}).get('totalDebt/totalEquityAnnual'),
    'current_ratio': metrics.get('metric', {}).get('currentRatioAnnual'),
    'revenue_growth': metrics.get('metric', {}).get('revenueGrowthTTM'),
    'fcf_margin': metrics.get('metric', {}).get('freeCashFlowMarginTTM')
}

fundamental_df = pd.DataFrame.from_dict(fundamental_data, orient='index', columns=[ticker])
fundamental_csv_path = os.path.join("data/fundamental_data", f"{ticker}_fundamental_data.csv")
fundamental_df.to_csv(fundamental_csv_path)

# data preprocessing
price_data = pd.read_csv("data/price_data/AAPL_price_data.csv", index_col=0, parse_dates=True)
fundamental_data = pd.read_csv("data/fundamental_data/AAPL_fundamental_data.csv", index_col=0).T

# Calculate technical indicators
price_data['50_MA'] = price_data['Close'].rolling(window=50).mean()
price_data['200_MA'] = price_data['Close'].rolling(window=200).mean()

# relative strength index
delta = price_data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss
price_data['RSI'] = 100 - (100 / (1 + rs))

# Bollinger Bands (Upper and Lower Bound)
rolling_mean = price_data['Close'].rolling(window=20).mean()
rolling_std = price_data['Close'].rolling(window=20).std()
price_data['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
price_data['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

# MACD (Moving Average Convergence Divergence)
short_ema = price_data['Close'].ewm(span=12, adjust=False).mean()
long_ema = price_data['Close'].ewm(span=26, adjust=False).mean()
price_data['MACD'] = short_ema - long_ema
price_data['MACD_Signal'] = price_data['MACD'].ewm(span=9, adjust=False).mean()

price_data.to_csv(price_csv_path)
fundamental_data.to_csv(fundamental_csv_path)
