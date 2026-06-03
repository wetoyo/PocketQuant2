import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from scraper.api_clients.YFinance import StockScraper
from scraper.utils.build_db import write_data_to_db

def main():
    print("Starting fetch of VIX and SPY daily data from 1990 to present...")
    tickers = ["^VIX", "SPY"]
    
    # We use 1990-01-01 as VIX was introduced in 1993, but yfinance has data starting from Jan 1990.
    start_date = "1990-01-01"
    end_date = "2026-06-02"
    
    scraper = StockScraper(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        adjusted=True,
        fill_missing=True
    )
    
    print("Fetching data from yfinance...")
    scraper.fetch_data()
    print("Cleaning data...")
    scraper.clean_data()
    
    # Verify data counts
    for ticker in tickers:
        df = scraper.data.get(ticker)
        if df is not None and not df.empty:
            print(f"Ticker {ticker}: fetched {len(df)} rows, from {df['DATE'].min()} to {df['DATE'].max()}")
        else:
            print(f"Warning: Ticker {ticker} has no data!")
            
    print("Saving to market_data.db...")
    write_data_to_db(scraper.data, db_path="market_data.db", interval="1d")
    print("Fetch and store completed successfully.")

if __name__ == "__main__":
    main()
