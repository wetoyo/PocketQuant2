import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from scraper.api_clients.YFinance import StockScraper
from configs.paths import DATA_RAW  # not used for Dynamo, but imported for consistency


# Setup logging
log_dir = current_dir / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Handlers
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
        logger.error(f"Error reading tickers file: {e}")
        return []


def table_near_limit(table_name, dynamo_client, limit_gb=24):
    """
    Check if DynamoDB table is near free tier limit (approx 25GB)
    """
    try:
        desc = dynamo_client.describe_table(TableName=table_name)
        size_bytes = desc['Table']['TableSizeBytes']
        size_gb = size_bytes / (1024 ** 3)
        if size_gb >= limit_gb:
            logger.error(f"Table {table_name} size {size_gb:.2f} GB exceeds limit {limit_gb} GB.")
            return True
        return False
    except ClientError as e:
        raise RuntimeError(f"Cannot check table size due to missing permissions: {e}")


def flatten_options_data(options_data):
    """
    Flatten nested options_data dict into a list of dicts
    """
    flattened = []
    for ticker, expirations in options_data.items():
        for exp, chains in expirations.items():
            for kind, df in chains.items():  # 'calls' or 'puts'
                for _, row in df.iterrows():
                    rec = row.to_dict()
                    rec.update({
                        "ticker": ticker,
                        "expiration": exp,
                        "type": kind,
                        "ticker_contract": f"{ticker}_{kind}_{exp}_{rec.get('strike')}",
                        "date": rec.get("FETCH_DATE") or str(pd.Timestamp.now().date())
                    })
                    flattened.append(rec)
    return flattened


def write_options_to_dynamo(options_data, table_name="options_data", batch_size=25):
    """
    Write flattened options data to DynamoDB
    - Prevent duplicates via ConditionExpression
    - Batch writes for speed
    """
    dynamo = boto3.client("dynamodb", region_name="us-east-1")

    if table_near_limit(table_name, dynamo):
        raise RuntimeError("Table is near limit. Aborting write to prevent charges.")

    records = flatten_options_data(options_data)
    if not records:
        logger.warning("No options records to write.")
        return

    logger.info(f"Writing {len(records)} options records to DynamoDB table '{table_name}'...")

    batch = []
    for rec in records:
        item = {k: {"S": str(v)} for k, v in rec.items() if v is not None}
        batch.append(item)

        if len(batch) == batch_size:
            _batch_write(dynamo, table_name, batch)
            batch = []

    if batch:
        _batch_write(dynamo, table_name, batch)

    logger.info("All records processed.")


def _batch_write(dynamo, table_name, items):
    """
    Batch write items to DynamoDB with duplicate prevention
    """
    for item in items:
        try:
            dynamo.put_item(
                TableName=table_name,
                Item=item,
                ConditionExpression="attribute_not_exists(date) AND attribute_not_exists(ticker_contract)"
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                logger.debug(f"Duplicate detected, skipping: {item['ticker_contract']['S']}")
            else:
                logger.error(f"Failed to write record {item.get('ticker_contract', {}).get('S')}: {e}")
                raise


def main():
    start_time = time.time()
    logger.info("Starting options fetch process...")

    tickers_file = current_dir / "tickers.txt"
    if not tickers_file.exists():
        logger.error(f"Tickers file not found at {tickers_file}")
        return

    tickers = read_tickers(tickers_file)
    if not tickers:
        logger.error("No tickers found in file.")
        return

    logger.info(f"Found {len(tickers)} tickers: {tickers}")

    try:
        # Fetch options only
        scraper = StockScraper(
            tickers=tickers,
            start_date=str(pd.Timestamp.now().date()),  # Only today
            end_date=str(pd.Timestamp.now().date() + pd.Timedelta(days=1)),
            interval="1d",
            adjusted=True,
            fill_missing=True
        )

        logger.info("Fetching options data...")
        scraper.fetch_options()
        logger.info("Writing options data to DynamoDB...")
        write_options_to_dynamo(scraper.options_data)
        logger.info("Options data saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
        sys.exit(1)  # Make GitHub Actions fail if error occurs

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Total execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
