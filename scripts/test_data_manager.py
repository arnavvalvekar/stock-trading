from data_manager import DataManager
import logging
import sys

def test_data_manager():
    # Initialize data manager
    dm = DataManager()
    
    # Test ticker
    ticker = "AAPL"
    
    try:
        # Test data fetching
        print(f"\nFetching data for {ticker}...")
        price_data, fundamental_data = dm.get_stock_data(ticker)
        
        # Print data info
        print("\nPrice Data Info:")
        print(f"Shape: {price_data.shape}")
        print(f"Columns: {price_data.columns.tolist()}")
        print(f"Date Range: {price_data.index.min()} to {price_data.index.max()}")
        
        print("\nFundamental Data Info:")
        print(f"Shape: {fundamental_data.shape}")
        print(f"Columns: {fundamental_data.columns.tolist()}")
        
        # Test data freshness
        print("\nData Freshness Info:")
        freshness = dm.get_data_freshness(ticker)
        print(f"Price Data Last Update: {freshness['price_data']['last_update']}")
        print(f"Fundamental Data Last Update: {freshness['fundamental_data']['last_update']}")
        
        # Test available tickers
        print("\nAvailable Tickers:")
        print(dm.get_available_tickers())
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_data_manager() 