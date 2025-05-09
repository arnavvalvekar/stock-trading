import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Tuple
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.price_dir = self.data_dir / "price_data"
        self.fundamental_dir = self.data_dir / "fundamental_data"
        self.metadata_dir = self.data_dir / "metadata"
        
        # Create necessary directories
        for directory in [self.price_dir, self.fundamental_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self.metadata_file = self.metadata_dir / "data_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata about data freshness and last updates."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata about data freshness and last updates."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def _is_data_fresh(self, ticker: str, data_type: str, max_age_hours: int = 24) -> bool:
        """Check if the data for a given ticker is fresh enough."""
        if ticker not in self.metadata:
            return False
        
        last_update = datetime.fromisoformat(self.metadata[ticker][data_type]['last_update'])
        age = datetime.now() - last_update
        return age.total_seconds() < (max_age_hours * 3600)

    def _validate_price_data(self, df: pd.DataFrame) -> bool:
        """Validate price data for required columns and data quality."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in price data: {required_columns}")
            return False
        
        if df.isnull().sum().sum() > 0:
            logger.warning("Price data contains null values")
            return False
        
        return True

    def _validate_fundamental_data(self, df: pd.DataFrame) -> bool:
        """Validate fundamental data for required fields."""
        required_fields = ['marketCap', 'trailingPE', 'trailingEps']
        
        if not all(field in df.columns for field in required_fields):
            logger.error(f"Missing required fields in fundamental data: {required_fields}")
            return False
        
        return True

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for price data."""
        # Moving Averages
        df['50_MA'] = df['Close'].rolling(window=50).mean()
        df['200_MA'] = df['Close'].rolling(window=200).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
        df['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

        # MACD
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = short_ema - long_ema
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        return df

    def get_stock_data(self, ticker: str, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get price and fundamental data for a stock, updating if necessary.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Force data refresh regardless of freshness
            
        Returns:
            Tuple of (price_data, fundamental_data) DataFrames
        """
        ticker = ticker.upper()
        price_path = self.price_dir / f"{ticker}_price_data.csv"
        fundamental_path = self.fundamental_dir / f"{ticker}_fundamental_data.csv"

        try:
            # Check if we need to update price data
            if force_refresh or not self._is_data_fresh(ticker, 'price_data'):
                logger.info(f"Fetching fresh price data for {ticker}")
                stock_data = yf.download(ticker, 
                                      start=(datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d'),
                                      end=datetime.now().strftime('%Y-%m-%d'),
                                      interval="1d")
                
                if not self._validate_price_data(stock_data):
                    raise ValueError(f"Invalid price data for {ticker}")
                
                stock_data = self._calculate_technical_indicators(stock_data)
                stock_data.to_csv(price_path)
                
                # Update metadata
                self.metadata[ticker] = {
                    'price_data': {
                        'last_update': datetime.now().isoformat(),
                        'rows': len(stock_data)
                    }
                }
                self._save_metadata()

            # Check if we need to update fundamental data
            if force_refresh or not self._is_data_fresh(ticker, 'fundamental_data'):
                logger.info(f"Fetching fresh fundamental data for {ticker}")
                stock = yf.Ticker(ticker)
                info = stock.info
                fundamental_df = pd.DataFrame.from_dict(info, orient="index", columns=[ticker])
                
                if not self._validate_fundamental_data(fundamental_df):
                    raise ValueError(f"Invalid fundamental data for {ticker}")
                
                fundamental_df.to_csv(fundamental_path)
                
                # Update metadata
                if ticker not in self.metadata:
                    self.metadata[ticker] = {}
                self.metadata[ticker]['fundamental_data'] = {
                    'last_update': datetime.now().isoformat(),
                    'rows': len(fundamental_df)
                }
                self._save_metadata()

            # Load the data
            price_data = pd.read_csv(price_path, index_col=0, parse_dates=True)
            fundamental_data = pd.read_csv(fundamental_path, index_col=0)

            return price_data, fundamental_data

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise

    def get_available_tickers(self) -> list:
        """Get list of tickers with available data."""
        return list(self.metadata.keys())

    def get_data_freshness(self, ticker: str) -> Dict:
        """Get information about data freshness for a ticker."""
        if ticker not in self.metadata:
            return {"status": "No data available"}
        
        return {
            "price_data": {
                "last_update": self.metadata[ticker]['price_data']['last_update'],
                "rows": self.metadata[ticker]['price_data']['rows']
            },
            "fundamental_data": {
                "last_update": self.metadata[ticker]['fundamental_data']['last_update'],
                "rows": self.metadata[ticker]['fundamental_data']['rows']
            }
        } 