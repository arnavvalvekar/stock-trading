# fetch economic data for macro model

import os
import pandas as pd
from fredapi import Fred
from datetime import datetime

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

INDICATORS = {
    'CPI': 'CPIAUCSL',                    # Consumer Price Index
    'Unemployment': 'UNRATE',            # Unemployment Rate
    'FedFundsRate': 'FEDFUNDS',          # Federal Funds Rate
    'RetailSales': 'RSAFS',              # Retail and Food Services Sales
    'IndustrialProduction': 'INDPRO',    # Industrial Production Index
    'LeadingIndicators': 'USSLIND'       # Leading Index for the United States
}

START_DATE = "2010-01-01"

def fetch_data():
    all_data = pd.DataFrame()

    for name, series_id in INDICATORS.items():
        print(f"Fetching {name}")
        series = fred.get_series(series_id, observation_start=START_DATE)
        df = series.to_frame(name)
        df.index = pd.to_datetime(df.index)
        all_data = all_data.join(df, how="outer") if not all_data.empty else df

    all_data = all_data.sort_index().ffill()
    return all_data

def main():
    macro_df = fetch_data()
    macro_df.to_csv("data/macro_data/macro_features.csv", index_label="Date")

if __name__ == "__main__":
    main()
