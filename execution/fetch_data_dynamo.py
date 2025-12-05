import sys
import logging
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from scraper.api_clients.YFinance import StockScraper

# Setup logging
log_dir = current_dir / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
stream_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

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

def table_near_limit(table_name, dynamo, threshold=0.9):
    """
    Checks if the table is near its provisioned limit (25GB free tier).
    """
    try:
        desc = dynamo.describe_table(TableName=table_name)
        size_bytes = desc['Table']['TableSizeBytes']
        max_bytes = 24 * 1024**3  # 25 GB free tier
        if size_bytes >= max_bytes * threshold:
            logging.error(f"Table {table_name} size {size_bytes} bytes exceeds threshold {threshold*100}%")
            return True
        return False
    except ClientError as e:
        logging.error(f"Cannot check table size due to missing permissions: {e}")
        raise RuntimeError(f"Cannot check table size due to missing permissions: {e}")

def flatten_options_data(options_data):
    """
    Convert nested options_data dict into a list of flat dicts.
    """
    records = []
    for ticker, dates_dict in options_data.items():
        for date, chains in dates_dict.items():
            for kind, df in chains.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                for _, row in df.iterrows():
                    # Partition key
                    fetch_date = row.get("FETCH_DATE")
                    if fetch_date is None:
                        fetch_date = pd.Timestamp.now().strftime("%Y-%m-%d")
                    
                    # Sort key
                    strike = row.get("Strike") or row.get("STRIKE") or "0"
                    ticker_contract = f"{ticker}_{date}_{kind}_{strike}"
                    
                    rec = row.to_dict()
                    rec.update({
                        "ticker": ticker,
                        "expiration": date,
                        "type": kind,
                        "date": str(fetch_date),
                        "ticker_contract": ticker_contract
                    })
                    records.append(rec)
    return records

def write_options_to_dynamo(options_data, table_name="options_data"):
    """
    Write flattened options data to DynamoDB with safety check.
    """
    dynamo = boto3.client("dynamodb", region_name="us-east-1")
    
    if table_near_limit(table_name, dynamo):
        raise RuntimeError("Table is near limit. Aborting write to prevent charges.")

    records = flatten_options_data(options_data)
    
    if not records:
        logging.warning("No options records to write.")
        return

    logging.info(f"Writing {len(records)} options records to DynamoDB table '{table_name}'...")

    for rec in records:
        item = {k: {"S": str(v)} for k, v in rec.items()}
        try:
            dynamo.put_item(TableName=table_name, Item=item)
        except Exception as e:
            logging.error(f"Failed to write record for {rec.get('ticker')}, expiration {rec.get('expiration')}: {e}")
            raise

def main():
    start_time = time.time()
    logging.info("Starting options fetch process...")

    try:
        tickers_file = current_dir / "tickers.txt"
        if not tickers_file.exists():
            raise FileNotFoundError(f"Tickers file not found at {tickers_file}")

        tickers = read_tickers(tickers_file)
        if not tickers:
            raise ValueError("No tickers found in file.")

        logging.info(f"Found {len(tickers)} tickers: {tickers}")

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

        logging.info("Fetching options data...")
        scraper.fetch_options()

        logging.info("Writing options data to DynamoDB...")
        write_options_to_dynamo(scraper.options_data)
        logging.info("Options data saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)
        sys.exit(1)

    duration = time.time() - start_time
    logging.info(f"Total execution time: {duration:.2f} seconds")

if __name__ == "__main__":
    main()
