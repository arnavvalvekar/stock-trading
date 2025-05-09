# Stock Trading AI Documentation

## Overview
This project implements an AI-based stock trading system that combines multiple machine learning models for market analysis and trading decisions. The system includes time series forecasting, stock valuation, and sentiment analysis components.

## Features
- Time series forecasting using CNN models
- Stock valuation analysis
- Sentiment analysis of market news
- Model management and versioning
- Configuration management
- Comprehensive error handling
- Logging and monitoring

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd stock-trading-ai
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
ALPHA_VANTAGE_API_KEY=your_api_key
MODEL_SAVE_DIR=models/saved
DATA_DIR=data
```

## Project Structure
```
stock_trading_ai/
├── config/
│   ├── __init__.py
│   └── config.py
├── data/
│   └── raw/
├── models/
│   ├── __init__.py
│   ├── model_manager.py
│   └── architectures.py
├── scripts/
│   └── test_model_manager.py
├── utils/
│   ├── __init__.py
│   └── error_handling.py
├── docs/
│   └── README.md
├── tests/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Usage

### Basic Usage
```python
from stock_trading_ai.models import ModelManager
from stock_trading_ai.config import Config

# Initialize configuration
config = Config()

# Create model manager
model_manager = ModelManager(config)

# Train a model
model_manager.train_model(
    model_name="cnn_forecast",
    data=train_data,
    epochs=100
)

# Make predictions
predictions = model_manager.predict(
    model_name="cnn_forecast",
    data=test_data
)
```

### Error Handling
The system includes comprehensive error handling:
```python
from stock_trading_ai.utils.error_handling import (
    retry_on_exception,
    handle_yfinance_errors,
    ErrorHandler
)

@retry_on_exception(max_attempts=3)
@handle_yfinance_errors
def fetch_stock_data(symbol):
    # Your code here
    pass

# Using context manager for cleanup
with ErrorHandler(cleanup_func=my_cleanup):
    # Your code here
    pass
```

## Configuration
The system uses a hierarchical configuration system that supports:
- Environment variables
- Configuration files
- Default values

Example configuration:
```python
config = {
    "api": {
        "alpha_vantage": {
            "api_key": "your_key",
            "base_url": "https://www.alphavantage.co/query"
        }
    },
    "models": {
        "cnn": {
            "input_size": 10,
            "num_classes": 3,
            "learning_rate": 0.001
        }
    }
}
```

## Testing
Run tests using pytest:
```bash
pytest tests/
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions and support, please open an issue in the repository. 