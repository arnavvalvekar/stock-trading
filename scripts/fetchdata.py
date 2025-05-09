# script to preprocess data for an individual stock

import os
import pandas as pd
import yfinance as yf

# csv creation
ticker = "AAPL"
start_date = "2015-01-01"
end_date = "2025-01-01"

# price data file
stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
price_csv_path = os.path.join("data/price_data", f"{ticker}_price_data.csv")
stock_data.to_csv(price_csv_path)

# fundamental data file
stock = yf.Ticker(ticker)
info = stock.info
fundamental_df = pd.DataFrame.from_dict(info, orient="index", columns=[ticker])
fundamental_csv_path = os.path.join("data/fundamental_data", f"{ticker}_fundamental_data.csv")
fundamental_df.to_csv(fundamental_csv_path)



# data preprocessing
price_data = pd.read_csv("data/price_data/AAPL_price_data.csv")
fundamental_data = pd.read_csv("data/fundamental_data/AAPL_fundamental_data.csv", index_col=0).T

# clean price_data
price_data = price_data.drop(1).drop(0)
price_data = price_data.rename(columns={"Price": "Date"})

price_data['50_MA'] = price_data['Close'].rolling(window=50).mean()
price_data['200_MA'] = price_data['Close'].rolling(window=200).mean()

# relative strength index
price_data["Close"] = price_data["Close"].astype(float)
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

# fundamental data
fundamental_data['P/E Ratio'] = fundamental_data['trailingPE'] if 'trailingPE' in fundamental_data.columns else None
fundamental_data['Market Cap'] = fundamental_data['marketCap'] if 'marketCap' in fundamental_data.columns else None
fundamental_data['EPS'] = fundamental_data['trailingEps'] if 'trailingEps' in fundamental_data.columns else None

price_data.to_csv(price_csv_path)
fundamental_data.to_csv(fundamental_csv_path)
