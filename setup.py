from setuptools import setup, find_packages

setup(
    name="stock_trading_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "yfinance>=0.1.70",
        "requests>=2.26.0",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.9.0",
        "streamlit>=0.85.0",
        "python-dotenv>=0.19.0",
        "plotly>=5.3.0"
    ],
    python_requires=">=3.8",
    author="Stock Trading AI Team",
    author_email="your.email@example.com",
    description="An AI-powered stock trading system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-trading-ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 