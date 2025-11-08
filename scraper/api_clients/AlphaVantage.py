# stock_scraper_alpha_vantage.py
import os
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List
import requests
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StockScraperAV:
    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT = 5  # free tier: 5 requests per minute

    def __init__(
        self, 
        api_key: str,
        tickers: List[str], 
        start_date: str, 
        end_date: str, 
        interval: str = "1d", 
        adjusted: bool = False, 
        fill_missing: bool = True
    ):
        self.api_key = api_key
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.interval = interval
        self.adjusted = adjusted
        self.fill_missing = fill_missing
        self.data = {}
        self._request_times = []

    def _throttle(self):
        """Ensure no more than RATE_LIMIT requests per minute."""
        now = datetime.now()
        # Remove timestamps older than 60s
        self._request_times = [t for t in self._request_times if (now - t).seconds < 60]
        if len(self._request_times) >= self.RATE_LIMIT:
            wait_time = 60 - (now - self._request_times[0]).seconds + 1
            logging.info(f"Rate limit reached, sleeping for {wait_time} seconds...")
            time.sleep(wait_time)
            self._request_times = [t for t in self._request_times if (datetime.now() - t).seconds < 60]
        self._request_times.append(datetime.now())

    def fetch_data(self):
        for ticker in self.tickers:
            retries = 5
            backoff = 12
            while retries > 0:
                try:
                    logging.info(f"Fetching data for {ticker}")
                    self._throttle()
                    df = self._fetch_ticker_data(ticker)
                    if df.empty:
                        logging.warning(f"No data found for {ticker}, skipping.")
                        break
                    df.reset_index(inplace=True)
                    df.rename(columns=lambda x: x.upper(), inplace=True)
                    required_cols = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
                    if self.adjusted:
                        required_cols.append("ADJ CLOSE")
                        if "ADJ CLOSE" not in df.columns:
                            df["ADJ CLOSE"] = pd.NA
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

    def _fetch_ticker_data(self, ticker: str) -> pd.DataFrame:
        if self.interval.endswith("min"):
            func = "TIME_SERIES_INTRADAY"
            av_interval = self.interval
        elif self.interval == "1d":
            func = "TIME_SERIES_DAILY_ADJUSTED" if self.adjusted else "TIME_SERIES_DAILY"
            av_interval = None
        else:
            raise ValueError("Unsupported interval for Alpha Vantage")

        params = {
            "function": func,
            "symbol": ticker,
            "apikey": self.api_key,
            "outputsize": "full",
        }
        if av_interval:
            params["interval"] = av_interval

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        data_json = response.json()

        if func.startswith("TIME_SERIES_INTRADAY"):
            key = f"Time Series ({av_interval})"
        elif func.startswith("TIME_SERIES_DAILY"):
            key = "Time Series (Daily)"
        else:
            key = None

        if key not in data_json:
            raise ValueError(f"No data returned for {ticker}: {data_json}")

        records = []
        for ts, values in data_json[key].items():
            record = {
                "DATE": pd.to_datetime(ts),
                "OPEN": float(values["1. open"]),
                "HIGH": float(values["2. high"]),
                "LOW": float(values["3. low"]),
                "CLOSE": float(values["4. close"]),
                "VOLUME": int(values["5. volume"]),
            }
            if self.adjusted and func.endswith("DAILY_ADJUSTED"):
                record["ADJ CLOSE"] = float(values["5. adjusted close"])
            records.append(record)

        df = pd.DataFrame(records)
        df = df[(df["DATE"] >= self.start_date) & (df["DATE"] <= self.end_date)]
        df.sort_values("DATE", inplace=True)
        return df

    def clean_data(self):
        for ticker, df in self.data.items():
            df.drop_duplicates(inplace=True)
            df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None)
            if self.fill_missing:
                df.interpolate(method='time', inplace=True)
                df.fillna(method='bfill', inplace=True)
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
        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            combined_file = os.path.join(folder, f"combined.{format}")
            if format.lower() == "csv":
                combined_df.to_csv(combined_file, index=False)
            else:
                combined_df.to_parquet(combined_file, index=False)
            logging.info(f"Saved combined data to {combined_file}")

    def get_data(self, ticker: str):
        return self.data.get(ticker)

# Example usage
if __name__ == "__main__":
    API_KEY = "YOUR_ALPHA_VANTAGE_KEY"
    scraper = StockScraperAV(
        api_key=API_KEY,
        tickers=["AAPL", "MSFT"], 
        start_date="2023-01-01", 
        end_date="2023-10-01", 
        interval="1min", 
        adjusted=True, 
        fill_missing=True
    )
    scraper.fetch_data()
    scraper.clean_data()
    scraper.save_data(folder="stock_data_av_throttle", format="csv")
