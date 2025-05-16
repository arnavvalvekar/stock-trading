#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import torch
import numpy as np
from datetime import datetime, timedelta, date
import finnhub
import os
from dotenv import load_dotenv
import logging
from functools import lru_cache
import yfinance as yf
from typing import Dict, Any

from stock_trading_ai.models import ModelManager
from stock_trading_ai.config import Config
from stock_trading_ai.config.logging_config import setup_logging, get_logger
from stock_trading_ai.models.forecast import load_forecast_model, get_cnn_signal
from stock_trading_ai.models.sentiment import get_sentiment_signal
from stock_trading_ai.models.technical_patterns import get_technical_signal
from stock_trading_ai.models.insider_model import get_insider_signal_wrapped as get_insider_signal
from stock_trading_ai.models.value_model import get_value_signal
from stock_trading_ai.models.macro_model import get_macro_signal

def display_model_signals_table(signals: Dict[str, tuple]):
    """Display model signals and confidences as a table."""
    data = []
    for model, (signal, confidence) in signals.items():
        data.append({
            "Model": model,
            "Signal": signal.upper(),
            "Confidence": f"{confidence:.1%}"
        })
    df = pd.DataFrame(data)
    st.table(df)


# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize PyTorch
torch.set_num_threads(1)  # Limit PyTorch to single thread to avoid conflicts

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Set up logging
setup_logging()
logger = get_logger("stock_trading_ai.app")

# Initialize Finnhub client
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
if not finnhub_api_key:
    st.error("Finnhub API key not found. Please check your .env file.")
    st.stop()
elif len(finnhub_api_key) < 10:  # Basic validation that key exists and has reasonable length
    st.error("Invalid Finnhub API key format. Please check your .env file.")
    st.stop()

finnhub_client = finnhub.Client(api_key=finnhub_api_key)

# Load forecast model
forecast_model = load_forecast_model(
    model_path=str(project_root / "models" / "forecast_model.pt"),
    input_size=1,
    seq_len=60,
    forecast_horizon=1
)

# Create data directory if it doesn't exist
data_dir = project_root / "data" / "cache"
data_dir.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=100)
def get_cached_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get cached stock data or fetch new data if not available.
    
    Args:
        symbol: Stock symbol
        start_date: Start date string
        end_date: End date string
        
    Returns:
        DataFrame with stock data
    """
    cache_file = data_dir / f"{symbol}_{start_date}_{end_date}.csv"
    
    # Check if cached data exists and is recent (less than 1 hour old)
    if cache_file.exists():
        file_age = time.time() - cache_file.stat().st_mtime
        if file_age < 3600:  # 1 hour in seconds
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Clean up any extra header rows
                df = df[~df.index.isin(['Price', 'Ticker', 'Date'])]
                # Ensure the data has the correct columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_columns):
                    return df
            except Exception as e:
                logger.warning(f"Error reading cache file: {e}")
    
    # If cache is invalid or doesn't exist, try to fetch new data
    try:
        # First try Finnhub
        try:
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            # Fetch data from Finnhub
            stock_data = finnhub_client.stock_candles(
                symbol,
                'D',  # Daily interval
                start_timestamp,
                end_timestamp
            )
            
            if stock_data and stock_data.get('s') == 'ok':
                # Create DataFrame with correct format
                df = pd.DataFrame({
                    'Open': stock_data['o'],
                    'High': stock_data['h'],
                    'Low': stock_data['l'],
                    'Close': stock_data['c'],
                    'Volume': stock_data['v']
                }, index=pd.to_datetime(stock_data['t'], unit='s'))
                
                # Save to cache
                df.to_csv(cache_file)
                return df
            else:
                logger.warning(f"Invalid data received from Finnhub for {symbol}")
        except Exception as e:
            logger.warning(f"Error fetching from Finnhub: {str(e)}")
        
        # If Finnhub fails, try yfinance as fallback
        logger.info(f"Trying yfinance as fallback for {symbol}")
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if not df.empty:
            # Ensure column names match
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            
            # Clean up any extra header rows
            df = df[~df.index.isin(['Price', 'Ticker', 'Date'])]
            
            # Save to cache
            df.to_csv(cache_file)
            return df
        else:
            logger.error(f"No data available from yfinance for {symbol}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def predict_stock(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """Make predictions using the model.
    
    Args:
        model: Trained model
        X: Input features
        
    Returns:
        Model predictions
    """
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        pred = model(X_tensor)
        return pred.numpy().flatten()

def get_sentiment_signal(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Get trading signal from sentiment analysis."""
    try:
        from stock_trading_ai.models.sentiment import get_sentiment_signal as get_sentiment
        return get_sentiment(data, symbol)
    except Exception as e:
        logger.error(f"Error getting sentiment signal: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "sentiment": {
                "overall": 0.0,
                "news": [],
                "insider": []
            }
        }

def get_insider_signal(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Get trading signal from insider trading data."""
    try:
        from stock_trading_ai.models.insider_model import get_insider_signal_wrapped as get_insider
        return get_insider(symbol)
    except Exception as e:
        logger.error(f"Error getting insider signal: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "insider_data": []
        }

def get_technical_signal(data: pd.DataFrame) -> Dict[str, Any]:
    """Get trading signal from technical analysis."""
    try:
        from stock_trading_ai.models.technical_patterns import get_technical_signal as get_technical
        return get_technical(data)
    except Exception as e:
        logger.error(f"Error getting technical signal: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "indicators": {}
        }

def get_cnn_signal(data: pd.DataFrame) -> Dict[str, Any]:
    """Get trading signal from CNN model."""
    try:
        from stock_trading_ai.models.forecast import get_cnn_signal as get_cnn
        return get_cnn(data)
    except Exception as e:
        logger.error(f"Error getting CNN signal: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "prediction": 0.0
        }

def get_combined_signal(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Get combined trading signal from all models."""
    try:
        # Get signals from each model
        sentiment = get_sentiment_signal(data, symbol)
        insider = get_insider_signal(data, symbol)
        technical = get_technical_signal(data)
        cnn = get_cnn_signal(data)
        value = get_value_signal(symbol)
        macro = get_macro_signal()
        
        # Combine signals with weights
        signals = {
            "Sentiment": (sentiment["signal"], sentiment["confidence"] * 0.15),
            "Insider": (insider["signal"], insider["confidence"] * 0.15),
            "Technical": (technical["signal"], technical["confidence"] * 0.2),
            "CNN": (cnn["signal"], cnn["confidence"] * 0.2),
            "Value": (value["signal"], value["confidence"] * 0.15),
            "Macro": (macro["signal"], macro["confidence"] * 0.15)
        }
        
        # Calculate weighted signal
        buy_score = sum(weight for signal, weight in signals.values() if signal == "buy")
        sell_score = sum(weight for signal, weight in signals.values() if signal == "sell")
        
        if buy_score > sell_score:
            signal = "buy"
            confidence = buy_score
        elif sell_score > buy_score:
            signal = "sell"
            confidence = sell_score
        else:
            signal = "hold"
            confidence = 0.5
            
        return {
            "signal": signal,
            "confidence": confidence,
            "model_signals": signals
        }
        
    except Exception as e:
        logger.error(f"Error getting combined signal: {str(e)}")
        return {
            "signal": "hold",
            "confidence": 0.5,
            "model_signals": {}
        }

def plot_model_signals(signals: Dict[str, tuple]):
    """Create a radar chart of model signals and their confidences."""
    import plotly.graph_objects as go

    models = list(signals.keys())
    confidences = [weight for _, weight in signals.values()]
    # Close the loop for radar chart
    models += [models[0]]
    confidences += [confidences[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=confidences,
        theta=models,
        fill='toself',
        name='Model Confidences',
        line=dict(color='royalblue')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Model Signal Confidences (Radar Chart)"
    )

    st.plotly_chart(fig, use_container_width=True)

def load_data(symbol: str, start_date: str | datetime | date, end_date: str | datetime | date) -> pd.DataFrame:
    """Load stock data for the given symbol and date range.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (string in YYYY-MM-DD format, datetime object, or date object)
        end_date: End date (string in YYYY-MM-DD format, datetime object, or date object)
        
    Returns:
        DataFrame with stock data
    """
    try:
        # Convert dates to string format for caching
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, (datetime, date)):
            end_date = end_date.strftime('%Y-%m-%d')
            
        # Get data from cache or fetch new
        df = get_cached_data(symbol, start_date, end_date)
        
        if df.empty:
            st.warning(f"No data found for {symbol}")
            return pd.DataFrame()
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def plot_stock_data(data: pd.DataFrame, symbol: str):
    """Create an interactive plot of stock data."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='OHLC'
    ))
    
    # Add volume bar chart
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Stock Trading AI",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Stock Trading AI")
    st.write("AI-powered stock analysis and trading system")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Symbol input
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a valid stock symbol (e.g., AAPL, GOOGL, MSFT)"
    )
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    start_date = col1.date_input("Start Date", value=start_date)
    end_date = col2.date_input("End Date", value=end_date)
    
    # Load configuration
    config = Config()
    
    # Initialize model manager
    model_manager = ModelManager(config)
    
    # Main content
    if st.sidebar.button("Analyze"):
        with st.spinner("Loading data and running analysis..."):
            # Load data
            data = load_data(symbol, start_date, end_date)
            
            if not data.empty:
                logger.info(f"Loaded data shape: {data.shape}")
                logger.info(f"Data columns: {data.columns.tolist()}")
                logger.info(f"Data sample:\n{data.head()}")
                
                # Display stock data
                st.subheader("Stock Data")
                st.dataframe(data)
                
                # Plot stock data
                st.subheader("Stock Chart")
                fig = plot_stock_data(data, symbol)
                st.plotly_chart(fig, use_container_width=True)
                
                # Run all models
                st.subheader("AI Analysis")
                
                try:
                    # Get combined signal
                    signal = get_combined_signal(data, symbol)
                    logger.info(f"Generated signal: {signal}")
                    
                    # Display final prediction
                    st.subheader("Final Prediction")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Signal", signal["signal"].upper())
                    with col2:
                        st.metric("Confidence", f"{signal['confidence']*100:.1f}%")

                    # Display model signals graph
                    # Display model signals table
                    st.subheader("Model Signals (Table View)")
                    display_model_signals_table(signal["model_signals"])
                    
                    # Display individual model signals
                    st.subheader("Model Details")
                    for model, (model_signal, confidence) in signal["model_signals"].items():
                        st.write(f"{model.title()}: {model_signal.upper()} ({confidence:.1%})")
                    
                    # Display additional information
                    st.subheader("Additional Information")
                    
                    # Technical indicators
                    if 'indicators' in signal['model_signals']['Technical']:
                        st.markdown("**Technical Indicators:**")
                        for indicator, value in signal['model_signals']['Technical']['indicators'].items():
                            st.markdown(f"- {indicator}: {value:.2f}")
                            
                    # Sentiment analysis
                    if 'sentiment' in signal['model_signals']['Sentiment']:
                        st.markdown("**Sentiment Analysis:**")
                        sentiment = signal['model_signals']['Sentiment']['sentiment']
                        st.markdown(f"- Overall Sentiment: {sentiment['overall']:.2f}")
                        if sentiment['news']:
                            st.markdown("**Recent News:**")
                            for article in sentiment['news'][:5]:  # Show top 5 news items
                                st.markdown(f"- {article['title']} ({article['sentiment']:.2f})")
                                
                    # Insider trading
                    if 'insider_data' in signal['model_signals']['Insider']:
                        st.markdown("**Insider Trading:**")
                        for transaction in signal['model_signals']['Insider']['insider_data'][:5]:  # Show top 5 transactions
                            st.markdown(f"- {transaction['name']}: {transaction['type']} ({transaction['shares']} shares)")
                    
                    st.success("Analysis complete!")
                    
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error(f"Error during analysis: {str(e)}")
            else:
                st.error("No data available for the selected symbol and date range.")

if __name__ == "__main__":
    main()
