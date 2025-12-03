import sys
import os
import logging
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from scraper.api_clients.YFinance import StockScraper
from configs.paths import DATA_RAW
from scraper.utils.build_db import write_data_to_db, write_options_to_db

# Setup logging
log_dir = current_dir / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler(log_file)
stream_handler = logging.StreamHandler(sys.stdout)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to logger
# Remove existing handlers to avoid duplicate logs if YFinance added some
if logger.hasHandlers():
    logger.handlers.clear()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def read_tickers(file_path):
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except Exception as e:
        logging.error(f"Error reading tickers file: {e}")
        return []

def main():
    start_time = time.time()
    logging.info("Starting data fetch process...")

    tickers_file = current_dir / "tickers.txt"
    if not tickers_file.exists():
        logging.error(f"Tickers file not found at {tickers_file}")
        return

    tickers = read_tickers(tickers_file)
    if not tickers:
        logging.error("No tickers found in file.")
        return

    logging.info(f"Found {len(tickers)} tickers: {tickers}")

    try:
        # Fetching only today's data
        today = datetime.now().date()
        start_date = today.strftime("%Y-%m-%d")
        end_date = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        scraper = StockScraper(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
            adjusted=True,
            fill_missing=True
        )

        # Fetch Stock Data
        logging.info("Fetching stock price data...")
        scraper.fetch_data()
        scraper.clean_data()
        # scraper.save_data(folder=str(DATA_RAW), format="csv")
        logging.info("Saving stock price data to database...")
        write_data_to_db(scraper.data, db_path="market_data.db")
        logging.info("Stock price data saved to database.")

        # Fetch Options Data
        logging.info("Fetching options data...")
        scraper.fetch_options()
        # scraper.save_options(folder=str(DATA_RAW), format="csv")
        logging.info("Saving options data to database...")
        write_options_to_db(scraper.options_data, db_name="options_data.db")
        logging.info("Options data saved to database.")

        logging.info("Data fetch process completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Total execution time: {duration:.2f} seconds")

if __name__ == "__main__":
    main()
