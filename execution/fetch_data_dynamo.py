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

# -----------------------------------
# LOGGING SETUP
# -----------------------------------
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

# -----------------------------------
# UTILS
# -----------------------------------
def read_tickers(file_path):
    try:
        with open(file_path, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except Exception as e:
        logging.error(f"Error reading tickers file: {e}")
        return []


def normalize_value(v):
    """Convert values to DynamoDB-friendly types."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


# -----------------------------------
# SAFETY CHECK
# -----------------------------------
def table_near_limit(table_name, threshold_gb=24):
    """Return True if the table is at or above threshold (default 24GB)."""
    dynamo = boto3.client("dynamodb")
    try:
        desc = dynamo.describe_table(TableName=table_name)
        size_bytes = desc["Table"]["TableSizeBytes"]
        size_gb = size_bytes / (1024**3)
        if size_gb >= threshold_gb:
            logging.warning(f"DynamoDB table {table_name} size {size_gb:.2f}GB >= threshold {threshold_gb}GB")
            return True
        return False
    except ClientError as e:
        logging.error(f"Could not describe table {table_name}: {e}")
        return True  # fail-safe


def write_options_to_dynamo(options_data, table_name="options_data"):
    dynamo = boto3.resource("dynamodb")
    table = dynamo.Table(table_name)

    if table_near_limit(table_name):
        logging.error("Table is near or over limit. Aborting write to prevent charges.")
        return

    if hasattr(options_data, "to_dict"):
        options_data = options_data.to_dict(orient="records")

    for row in options_data:
        ticker = row.get("ticker")
        exp = row.get("expiration")
        strike = row.get("strike")
        opt_type = row.get("type")

        if not ticker or not exp or strike is None or not opt_type:
            logging.warning(f"Skipping malformed option row: {row}")
            continue

        option_key = f"{exp}_{strike}_{opt_type}"
        item = {"ticker": str(ticker), "option_key": option_key}

        for k, v in row.items():
            item[k] = normalize_value(v)

        try:
            table.put_item(Item=item)
        except ClientError as e:
            logging.error(f"Failed to write options row to DynamoDB: {e}")

def main():
    start_time = time.time()
    logging.info("Starting options fetch process...")

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

        # Fetch only options data
        logging.info("Fetching options data...")
        scraper.fetch_options()

        logging.info("Writing options data to DynamoDB...")
        write_options_to_dynamo(scraper.options_data)
        logging.info("Options data saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred during execution: {e}", exc_info=True)

    duration = time.time() - start_time
    logging.info(f"Total execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
