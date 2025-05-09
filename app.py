#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import torch
import numpy as np
from datetime import datetime, timedelta

# Initialize PyTorch
torch.set_num_threads(1)  # Limit PyTorch to single thread to avoid conflicts

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from stock_trading_ai.models import ModelManager
from stock_trading_ai.config import Config
from stock_trading_ai.config.logging_config import setup_logging, get_logger
from stock_trading_ai.models.ensemble import weighted_vote
from stock_trading_ai.models.forecast import load_forecast_model
from stock_trading_ai.models.value_model import get_value_signal
from stock_trading_ai.models.sentiment import analyze_stock_sentiment
from stock_trading_ai.models.technical_patterns import get_technical_signal
from stock_trading_ai.models.insider_model import get_insider_signal_wrapped as get_insider_signal
from stock_trading_ai.models.macro_model import get_macro_signal

# Set up logging
setup_logging()
logger = get_logger("stock_trading_ai.app")

def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load stock data for the given symbol and date range."""
    try:
        import yfinance as yf
        
        # Add delay between requests to avoid rate limiting
        time.sleep(1)
        
        # Use Ticker object for more reliable data fetching
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            st.warning(f"No data found for {symbol}")
            return pd.DataFrame()
            
        return data
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
    
    # Add moving averages
    if '50_MA' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['50_MA'],
            name='50-day MA',
            line=dict(color='blue')
        ))
    
    if '200_MA' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['200_MA'],
            name='200-day MA',
            line=dict(color='red')
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
        xaxis_rangeslider_visible=False
    )
    
    return fig

def plot_model_signals(signals: dict):
    """Create a radar chart of model signals."""
    fig = go.Figure()
    
    # Convert signals to numerical values
    signal_map = {"buy": 1, "hold": 0, "sell": -1}
    values = [signal_map[s["signal"]] for s in signals.values()]
    names = list(signals.keys())
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=names,
        fill='toself',
        name='Model Signals'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )
        ),
        showlegend=False
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
                    # Get signals from all models
                    signals = {
                        "CNN Forecast": get_cnn_signal(data),
                        "Value Analysis": get_value_signal(symbol),
                        "Sentiment": analyze_stock_sentiment(symbol),
                        "Technical": get_technical_signal(data),
                        "Insider": get_insider_signal(symbol),
                        "Macro": get_macro_signal()
                    }
                    
                    # Plot model signals
                    st.subheader("Model Signals")
                    signal_fig = plot_model_signals(signals)
                    st.plotly_chart(signal_fig, use_container_width=True)
                    
                    # Get ensemble prediction
                    final_signal, confidence = weighted_vote(signals)
                    
                    # Display results
                    st.subheader("Final Prediction")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Signal", final_signal.upper())
                    with col2:
                        st.metric("Confidence", f"{abs(confidence)*100:.1f}%")
                    
                    # Display individual model signals
                    st.subheader("Individual Model Signals")
                    for model_name, signal in signals.items():
                        st.write(f"{model_name}: {signal['signal'].upper()}")
                    
                    st.success("Analysis complete!")
                    
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error(f"Error during analysis: {str(e)}")
            else:
                st.error("No data available for the selected symbol and date range.")

if __name__ == "__main__":
    main()
