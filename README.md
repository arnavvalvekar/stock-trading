# Stock Trading AI

An AI-powered stock trading system that combines multiple machine learning models for market analysis and trading decisions.

## Features

- **Time Series Forecasting**: CNN-based models for predicting stock price movements
- **Stock Valuation**: Analysis of company fundamentals and market indicators
- **Sentiment Analysis**: Processing of news and social media for market sentiment
- **Model Management**: Versioning and tracking of model performance
- **Configuration Management**: Flexible configuration system for API keys and model parameters
- **Error Handling**: Robust error handling and logging system
- **Web Interface**: Streamlit-based user interface for easy interaction

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
LOG_DIR=logs
LOG_LEVEL=DEBUG
```

## Usage

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

### Using the API

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

## Project Structure

```
stock_trading_ai/
├── config/
│   ├── __init__.py
│   ├── config.py
│   └── logging_config.py
├── data/
│   └── raw/
├── models/
│   ├── __init__.py
│   ├── model_manager.py
│   └── architectures.py
├── scripts/
│   ├── setup_dev.py
│   └── test_model_manager.py
├── utils/
│   ├── __init__.py
│   └── error_handling.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_error_handling.py
│   ├── test_logging.py
│   └── test_model_manager.py
├── docs/
│   └── README.md
├── .env
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Development

### Setting Up Development Environment

Run the development setup script:
```bash
python scripts/setup_dev.py
```

### Running Tests

Run the test suite:
```bash
pytest tests/
```

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking
- isort for import sorting

Run the code quality checks:
```bash
black .
flake8
mypy .
isort .
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