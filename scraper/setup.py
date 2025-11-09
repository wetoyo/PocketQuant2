# setup.py

from pathlib import Path
import pandas as pd
import sqlite3

# Import scrapers
from api_clients.YFinance import StockScraper
from api_clients.AlphaVantage import StockScraperAV

# Import feature builder
from utils.feature_builder import build_features

# Import database helpers
from utils.build_db import create_or_connect_db, read_table

# Import project paths
from configs.paths import DATA_RAW, DATA_PROCESSED, DATA_FEATURES, DATABASE_PATH


class BaseSetup:
    def __init__(
        self,
        tickers,
        db_name="database.db",
        scraper_type="yfinance",  # "yfinance" or "alphavantage"
        api_key=None,
        start_date=None,
        end_date=None,
        interval="1d",
        adjusted=True,
        fill_missing=True,
        features_options=None,  # dict with MA, RSI, BB, etc.
    ):
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
        """Internal method to build features from data."""
        if df is None:
            df = self.data
        self.features = build_features(
            df=df,
            save_path=save_path,
            **self.features_options
        )
        return self.features

    def _write_to_db(self, table_name="features_table", append=True):
        """Internal method to write features to DB."""
        if self.features.empty:
            raise ValueError("No features available. Run _build_features first.")
        conn = self.connect_db()
        self.features.to_sql(
            table_name,
            conn,
            if_exists="append" if append else "replace",
            index=False
        )
        conn.close()
        print(f"Features written to table '{table_name}' in {self.db_path}")

    def _read_from_db(self, table_name="features_table"):
        """Read features from DB as pandas DataFrame."""
        df = read_table(self.db_path, table_name)
        return df

    def run_pipeline(
        self,
        save_folder_raw: Path = DATA_RAW,
        save_folder_features: Path = DATA_FEATURES,
        table_name="features_table"
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
        self._write_to_db(table_name=table_name, append=True)

        print("Reading features from database...")
        df_from_db = self._read_from_db(table_name=table_name)
        return df_from_db
    
    @property
    def df(self) -> pd.DataFrame:
        """Return the raw scraped data."""
        return self.data

    # Accessor for feature DataFrame
    @property
    def features_df(self) -> pd.DataFrame:
        """Return the processed features DataFrame."""
        return self.features