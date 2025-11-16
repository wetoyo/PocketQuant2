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
from scraper.utils.build_db import create_or_connect_db, read_table

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
    ):
        
        if db_name is None:
            tickers_sorted = sorted([t.upper() for t in tickers])
            db_name = f"{'_'.join(tickers_sorted)}.db"
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

        self.data: pd.DataFrame = pd.DataFrame()
        self.features: pd.DataFrame = pd.DataFrame()
    
    def connect_db(self):
        """Create or connect to database."""
        conn = create_or_connect_db(self.db_path)
        return conn

    def _fetch_data(self, save_folder: Path = DATA_RAW):
        """Internal method to fetch data using the selected scraper."""
        if self.scraper_type.lower() == "yfinance":
            scraper = StockScraper(
                tickers=self.tickers,
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
                tickers=self.tickers,
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

        if save_folder:
            scraper.save_data(folder=save_folder, format="csv")

        self.data = scraper.data
        return self.data

    def _build_features(self, df=None, save_path: Path = DATA_FEATURES):
        """
        Build features for EACH ticker separately.
        Returns a dict: { ticker : features_df }.
        """
        if df is None:
            df = self.data

        if not isinstance(df, dict):
            raise ValueError("Expected df to be a dict of DataFrames")

        features_by_ticker = {}

        for ticker, sub_df in df.items():

            if "DATE" not in sub_df.columns:
                raise KeyError(f"Ticker {ticker} is missing DATE column.")

            sub_df = sub_df.sort_values("DATE").reset_index(drop=True)

            features_df = build_features(
                df=sub_df,
                save_path=save_path,
                **self.features_options
            )

            features_df["TICKER"] = ticker

            features_by_ticker[ticker] = features_df

        # Store full dict
        self.features = features_by_ticker
        return self.features

    def _write_to_db(self, db_path="database.db"):
        """
        Write features to the database, one table per ticker.
        Updates existing rows based on DATE.
        """
        if not isinstance(self.features, dict) or not self.features:
            raise ValueError("Features must be a non-empty dict. Run _build_features first.")

        from scraper.utils.build_db import write_features_dict_to_db

        write_features_dict_to_db(self.features, db_path=db_path)
        print(f"All features written/updated in {db_path}")

    def _read_from_db(self, table_name="features_table"):
        """Read features from DB as pandas DataFrame."""
        df = read_table(self.db_path, table_name)
        return df

    def run_pipeline(
        self,
        save_folder_raw: Path = DATA_RAW,
        save_folder_features: Path = DATA_FEATURES,
        db_path: Path = None
    ):
        """
        Run the full pipeline:
        1. Scrape data
        2. Build features
        3. Write to DB
        4. Read back as DataFrame
        """
        print("Fetching data...")
        self._fetch_data(save_folder=save_folder_raw)

        print("Building features...")
        self._build_features(df=self.data, save_path=save_folder_features)

        print("Writing features to database...")
        print(self.db_path)
        self._write_to_db(db_path=self.db_path)

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