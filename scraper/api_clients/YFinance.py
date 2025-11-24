# yfinance.py
import os
import time
import logging
import sys
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List

root = Path(__file__).parent.parent.parent  # project_root
sys.path.append(str(root))
from configs.paths import DATA_RAW, DATA_PROCESSED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockScraper:
    def __init__(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str, 
        interval: str = "1d", 
        adjusted: bool = False, 
        fill_missing: bool = True
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.adjusted = adjusted
        self.fill_missing = fill_missing
        self.data = {}  # stores DataFrame per ticker

    def fetch_data(self):
        for ticker in self.tickers:
            retries = 3
            backoff = 1
            while retries > 0:
                try:
                    logging.info(f"Fetching data for {ticker}")
                    df = yf.download(
                        tickers=ticker,
                        start=self.start_date,
                        end=self.end_date,
                        interval=self.interval,
                        progress=False,
                        auto_adjust=self.adjusted,
                        threads=True
                    )
                    if df.empty:
                        logging.warning(f"No data found for {ticker}, skipping.")
                        break
                    df.reset_index(inplace=True)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.rename(columns=lambda x: x.upper(), inplace=True)
                    required_cols = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
                    if "ADJ CLOSE" in df.columns:
                        required_cols.append("ADJ CLOSE")
                    for col in required_cols:
                        if col not in df.columns:
                            df[col] = pd.NA
                    self.data[ticker] = df[required_cols]
                    break
                except Exception as e:
                    logging.error(f"Error fetching {ticker}: {e}")
                    retries -= 1
                    time.sleep(backoff)
                    backoff *= 2
            else:
                logging.warning(f"Failed to fetch data for {ticker} after retries.")

    def clean_data(self):
        for ticker, df in self.data.items():
            df.drop_duplicates(inplace=True)
            df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None)
            df.set_index("DATE", inplace=True)
            if self.fill_missing:
                df.interpolate(method='time', inplace=True)
                df.bfill(inplace=True)
            df.reset_index(inplace=True)
            self.data[ticker] = df

    def save_data(self, folder: str, format: str = "csv"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        combined = []

        for ticker, df in self.data.items():
            file_path = os.path.join(folder, f"{ticker}.{format}")
            if format.lower() == "csv":
                df.to_csv(file_path, index=False)
            elif format.lower() == "parquet":
                df.to_parquet(file_path, index=False)
            else:
                logging.warning(f"Unknown format {format}, skipping save for {ticker}")
                continue

            combined.append(df.assign(TICKER=ticker))
            logging.info(f"Saved {ticker} data to {file_path}")

        if len(combined) > 1:
            combined_df = pd.concat(combined, ignore_index=True)

            sorted_tickers = sorted(self.data.keys(), key=str.lower)
            ticker_filename = "_".join(t.upper() for t in sorted_tickers)

            combined_file = os.path.join(folder, f"{ticker_filename}.{format}")

            if format.lower() == "csv":
                combined_df.to_csv(combined_file, index=False)
            else:
                combined_df.to_parquet(combined_file, index=False)

            logging.info(f"Saved combined data to {combined_file}")

    def get_data(self, ticker: str):
        return self.data.get(ticker)


# Example usage
if __name__ == "__main__":
    scraper = StockScraper(
        tickers=["AAPL", "MSFT"], 
        start_date="2023-01-01", 
        end_date="2023-10-01", 
        interval="1d", 
        adjusted=True, 
        fill_missing=True
    )
    scraper.fetch_data()
    scraper.clean_data()
    scraper.save_data(folder=DATA_RAW, format="csv")
