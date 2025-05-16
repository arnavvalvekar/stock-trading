import os
import pandas as pd
import finnhub
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
        
        # Initialize Finnhub client
        self.finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

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

    def _validate_price_data(self, data: pd.DataFrame) -> bool:
        """Validate price data format and content."""
        required_columns = ['c', 'h', 'l', 'o', 'v']  # Finnhub OHLCV columns
        return all(col in data.columns for col in required_columns) and not data.empty

    def _validate_fundamental_data(self, data: pd.DataFrame) -> bool:
        """Validate fundamental data format and content."""
        required_columns = ['metric', 'value']
        return all(col in data.columns for col in required_columns) and not data.empty

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for price data."""
        # Rename columns to match standard format
        data = data.rename(columns={
            'c': 'Close',
            'h': 'High',
            'l': 'Low',
            'o': 'Open',
            'v': 'Volume'
        })
        
        # Calculate moving averages
        data['50_MA'] = data['Close'].rolling(window=50).mean()
        data['200_MA'] = data['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
        data['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
        
        # Calculate MACD
        short_ema = data['Close'].ewm(span=12, adjust=False).mean()
        long_ema = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = short_ema - long_ema
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        
        return data

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
                
                # Get current timestamp and 5 years ago
                end_timestamp = int(datetime.now().timestamp())
                start_timestamp = int((datetime.now() - timedelta(days=365*5)).timestamp())
                
                # Fetch candlestick data from Finnhub
                stock_data = self.finnhub_client.stock_candles(
                    ticker,
                    'D',  # Daily interval
                    start_timestamp,
                    end_timestamp
                )
                
                if not stock_data or stock_data.get('s') != 'ok':
                    raise ValueError(f"Invalid price data for {ticker}")
                
                # Convert to DataFrame
                df = pd.DataFrame({
                    'c': stock_data['c'],  # Close
                    'h': stock_data['h'],  # High
                    'l': stock_data['l'],  # Low
                    'o': stock_data['o'],  # Open
                    'v': stock_data['v']   # Volume
                }, index=pd.to_datetime(stock_data['t'], unit='s'))
                
                if not self._validate_price_data(df):
                    raise ValueError(f"Invalid price data for {ticker}")
                
                df = self._calculate_technical_indicators(df)
                df.to_csv(price_path)
                
                # Update metadata
                self.metadata[ticker] = {
                    'price_data': {
                        'last_update': datetime.now().isoformat(),
                        'rows': len(df)
                    }
                }
                self._save_metadata()

            # Check if we need to update fundamental data
            if force_refresh or not self._is_data_fresh(ticker, 'fundamental_data'):
                logger.info(f"Fetching fresh fundamental data for {ticker}")
                
                # Get company profile and financials
                profile = self.finnhub_client.company_profile2(symbol=ticker)
                metrics = self.finnhub_client.company_basic_financials(symbol=ticker, metric="all")
                
                # Combine data
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