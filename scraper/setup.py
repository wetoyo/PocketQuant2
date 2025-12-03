# setup.py

from pathlib import Path
import pandas as pd
import sqlite3


import sys
root = Path(__file__).parent.parent  # project_root
sys.path.append(str(root))

# Import scrapers
from scraper.api_clients.YFinance import StockScraper
from scraper.api_clients.AlphaVantage import StockScraperAV
# Import feature builder
from scraper.utils.feature_builder import build_features

# Import database helpers
from scraper.utils.build_db import (
    create_or_connect_db, 
    read_table, 
    write_data_to_db, 
    write_options_to_db,
    get_ticker_date_range, 
    read_by_date
)

# Import project paths
from configs.paths import DATA_RAW, DATA_PROCESSED, DATA_FEATURES, DATABASE_PATH


class BaseSetup:
    def __init__(
        self,
        tickers,
        db_name=None,
        scraper_type="yfinance",  # "yfinance" or "alphavantage"
        api_key=None,
        start_date=None,
        end_date=None,
        interval="1d",
        adjusted=True,
        fill_missing=True,
        features_options=None,  # dict with MA, RSI, BB, etc.
        include_options=False,
    ):
        
        if db_name is None:
            db_name = "market_data.db"
            
        self.tickers = tickers
        self.db_name = db_name
        
        self.db_path = DATABASE_PATH / db_name
        self.scraper_type = scraper_type
        self.api_key = api_key
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.adjusted = adjusted
        self.fill_missing = fill_missing
        self.include_options = include_options
        
        # Feature options with defaults
        self.features_options = features_options or {
            "returns": True,
            "log_returns": True,
            "ma_windows": [5, 20, 50],
            "bb_windows": [20],
            "rsi_windows": [14],
            "macd_params": [(12, 26, 9)],
            "vol_windows": [10],
            "atr_windows": [14]
        }

        self.data: pd.DataFrame = pd.DataFrame() # This might need to be a dict if multiple tickers
        self.data_dict = {} # Store raw data per ticker
        self.features: pd.DataFrame = pd.DataFrame()
    
    def connect_db(self):
        """Create or connect to database."""
        conn = create_or_connect_db(self.db_path)
        return conn

    def _fetch_data(self, save_folder: Path = DATA_RAW):
        """
        Fetch data for each ticker.
        Check DB first. If missing or incomplete, fetch from API and update DB.
        """
        self.data_dict = {}
    
        
        tickers_to_fetch = []
        
        for ticker in self.tickers:
            print(f"Checking data for {ticker}...")
            
            # Check DB
            min_date, max_date = get_ticker_date_range(self.db_path, ticker)
            
            data_needed = False
            if min_date is None or max_date is None:
                data_needed = True
                print(f"  No data found in DB for {ticker}.")
            else:
                db_min = pd.to_datetime(min_date)
                db_max = pd.to_datetime(max_date)
                
                req_start = pd.to_datetime(self.start_date) if self.start_date else None
                req_end = pd.to_datetime(self.end_date) if self.end_date else pd.Timestamp.now()
                
                if req_start and req_start < db_min:
                    data_needed = True
                    print(f"  Requested start {req_start.date()} < DB start {db_min.date()}. Fetching...")
                elif req_end and req_end > db_max + pd.Timedelta(days=1): # Allow 1 day buffer
                    data_needed = True
                    print(f"  Requested end {req_end.date()} > DB end {db_max.date()}. Fetching...")
                else:
                    print(f"  Data for {ticker} exists in DB ({db_min.date()} to {db_max.date()}). Loading from DB.")
                    # Load from DB
                    df = read_by_date(self.db_path, ticker, self.start_date, self.end_date)
                    self.data_dict[ticker] = df

            if data_needed:
                tickers_to_fetch.append(ticker)

        if tickers_to_fetch:
            print(f"Fetching from API for: {tickers_to_fetch}")
            if self.scraper_type.lower() == "yfinance":
                scraper = StockScraper(
                    tickers=tickers_to_fetch,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval=self.interval,
                    adjusted=self.adjusted,
                    fill_missing=self.fill_missing
                )
            elif self.scraper_type.lower() == "alphavantage":
                if not self.api_key:
                    raise ValueError("API key required for AlphaVantage scraper")
                scraper = StockScraperAV(
                    api_key=self.api_key,
                    tickers=tickers_to_fetch,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval=self.interval,
                    adjusted=self.adjusted,
                    fill_missing=self.fill_missing
                )
            else:
                raise ValueError("scraper_type must be 'yfinance' or 'alphavantage'")

            scraper.fetch_data()
            scraper.clean_data()
            
            # Update self.data_dict with new data
            # scraper.data is a dict {ticker: df}
            for ticker, df in scraper.data.items():
                self.data_dict[ticker] = df
            
            # Write new data to DB
            print("Writing fetched data to database...")
            write_data_to_db(scraper.data, db_path=self.db_path)
            
            # Optionally save to CSV raw if needed (legacy support or backup)
            if save_folder:
                scraper.save_data(folder=save_folder, format="csv")

        # Fetch options if requested (independent of price data)
        if self.include_options:
            print("Fetching options data...")
            # Create a scraper instance for all tickers to fetch options
            # We use the same configuration as the main scraper
            if self.scraper_type.lower() == "yfinance":
                options_scraper = StockScraper(
                    tickers=self.tickers,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval=self.interval,
                    adjusted=self.adjusted,
                    fill_missing=self.fill_missing
                )
                if hasattr(options_scraper, 'fetch_options'):
                    options_scraper.fetch_options()
                    # Save options to DB
                    print("Writing options data to database...")
                    write_options_to_db(options_scraper.options_data)
            else:
                print("Options fetching is only supported for yfinance scraper.")

        self.data = self.data_dict
        return self.data

    def _build_features(self, df=None, save_path: Path = DATA_FEATURES):
        """
        Build features for EACH ticker separately.
        Writes features to CSV files.
        Returns a dict: { ticker : features_df }.
        """
        if df is None:
            df = self.data

        if not isinstance(df, dict):
            # If it's a single DataFrame, wrap it? Or error?
            # Original code expected dict.
            raise ValueError("Expected df to be a dict of DataFrames")

        features_by_ticker = {}
        
        # Ensure save_path exists
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        for ticker, sub_df in df.items():
            if sub_df.empty:
                print(f"Skipping features for {ticker} (empty data)")
                continue

            if "DATE" not in sub_df.columns:
                # Try to fix if index is date
                if isinstance(sub_df.index, pd.DatetimeIndex):
                    sub_df = sub_df.reset_index()
                    if "Date" in sub_df.columns:
                        sub_df = sub_df.rename(columns={"Date": "DATE"})
                
                if "DATE" not in sub_df.columns:
                    print(f"Skipping {ticker}: Missing DATE column.")
                    continue

            sub_df = sub_df.sort_values("DATE").reset_index(drop=True)

            features_df = build_features(
                df=sub_df,
                save_path=None, # Don't save inside build_features, we save here
                **self.features_options
            )

            features_df["TICKER"] = ticker
            
            if save_path:
                # Save to CSV
                csv_file = save_path / f"{ticker}_features.csv"
                features_df.to_csv(csv_file, index=False)
                print(f"Saved features for {ticker} to {csv_file}")

            features_by_ticker[ticker] = features_df

        # Store full dict
        self.features = features_by_ticker
        return self.features

    def run_pipeline(
        self,
        save_folder_raw: Path = DATA_RAW,
        save_folder_features: Path = DATA_FEATURES,
    ):
        """
        Run the full pipeline:
        1. Fetch data (check DB, fetch API if needed, write to DB)
        2. Build features
        3. Write features to CSV
        """
        print("Starting pipeline...")
        self._fetch_data(save_folder=save_folder_raw)

        print("Building and saving features...")
        self._build_features(df=self.data, save_path=save_folder_features)

        print("Finished setup.")
        return self.features
    
    @property
    def df(self) -> pd.DataFrame:
        """Return the raw scraped data."""
        return self.data

    # Accessor for feature DataFrame
    @property
    def features_df(self) -> pd.DataFrame:
        """Return the processed features DataFrame."""
        return self.features