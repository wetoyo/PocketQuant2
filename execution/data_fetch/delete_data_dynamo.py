import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from collections import defaultdict

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from execution import LOGS_DIR


# Setup logging
log_file = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_delete.log"

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


def get_all_tickers(table_name="options_data"):
    """
    Scan DynamoDB table and return all unique tickers
    """
    dynamo = boto3.client("dynamodb", region_name="us-east-1")
    
    logger.info(f"Scanning DynamoDB table '{table_name}' for tickers...")
    
    tickers = set()
    try:
        # Use scan to get all items
        paginator = dynamo.get_paginator('scan')
        page_iterator = paginator.paginate(
            TableName=table_name,
            ProjectionExpression='ticker'
        )
        
        for page in page_iterator:
            for item in page.get('Items', []):
                if 'ticker' in item and 'S' in item['ticker']:
                    tickers.add(item['ticker']['S'])
        
        logger.info(f"Found {len(tickers)} unique tickers in the database.")
        return sorted(list(tickers))
    
    except ClientError as e:
        logger.error(f"Error scanning table: {e}")
        return []


def count_records_by_ticker(table_name="options_data"):
    """
    Count how many records exist for each ticker
    """
    dynamo = boto3.client("dynamodb", region_name="us-east-1")
    
    logger.info(f"Counting records by ticker in table '{table_name}'...")
    
    ticker_counts = defaultdict(int)
    try:
        paginator = dynamo.get_paginator('scan')
        page_iterator = paginator.paginate(
            TableName=table_name,
            ProjectionExpression='ticker'
        )
        
        for page in page_iterator:
            for item in page.get('Items', []):
                if 'ticker' in item and 'S' in item['ticker']:
                    ticker_counts[item['ticker']['S']] += 1
        
        return dict(ticker_counts)
    
    except ClientError as e:
        logger.error(f"Error counting records: {e}")
        return {}


def delete_ticker_records(ticker, table_name="options_data"):
    """
    Delete all records for a specific ticker from DynamoDB
    """
    dynamo = boto3.client("dynamodb", region_name="us-east-1")
    
    logger.info(f"Deleting all records for ticker '{ticker}' from table '{table_name}'...")
    
    deleted_count = 0
    try:
        # Scan for all items with this ticker
        paginator = dynamo.get_paginator('scan')
        page_iterator = paginator.paginate(
            TableName=table_name,
            FilterExpression='ticker = :ticker_val',
            ExpressionAttributeValues={':ticker_val': {'S': ticker}}
        )
        
        for page in page_iterator:
            for item in page.get('Items', []):
                # Delete each item using its primary key
                # Assuming 'date' and 'ticker_contract' are the primary keys
                if 'date' in item and 'ticker_contract' in item:
                    try:
                        dynamo.delete_item(
                            TableName=table_name,
                            Key={
                                'date': item['date'],
                                'ticker_contract': item['ticker_contract']
                            }
                        )
                        deleted_count += 1
                    except ClientError as e:
                        logger.error(f"Failed to delete item: {e}")
        
        logger.info(f"Successfully deleted {deleted_count} records for ticker '{ticker}'.")
        return deleted_count
    
    except ClientError as e:
        logger.error(f"Error deleting records for ticker '{ticker}': {e}")
        return 0


def main():
    start_time = time.time()
    logger.info("Starting DynamoDB ticker management process...")
    
    # Get all tickers
    tickers = get_all_tickers()
    
    if not tickers:
        logger.warning("No tickers found in the database.")
        return
    
    # Get record counts
    ticker_counts = count_records_by_ticker()
    
    # Display tickers with counts
    print("\n" + "="*60)
    print("TICKERS IN DATABASE")
    print("="*60)
    for i, ticker in enumerate(tickers, 1):
        count = ticker_counts.get(ticker, 0)
        print(f"{i:3d}. {ticker:10s} ({count:,} records)")
    print("="*60 + "\n")
    
    # Prompt user for deletion
    print("Enter ticker numbers to delete (comma-separated), or 'all' to delete all, or 'q' to quit:")
    user_input = input("> ").strip().lower()
    
    if user_input == 'q':
        logger.info("User cancelled operation.")
        return
    
    tickers_to_delete = []
    
    if user_input == 'all':
        tickers_to_delete = tickers
    else:
        try:
            indices = [int(x.strip()) for x in user_input.split(',')]
            for idx in indices:
                if 1 <= idx <= len(tickers):
                    tickers_to_delete.append(tickers[idx - 1])
                else:
                    logger.warning(f"Invalid index: {idx}")
        except ValueError:
            logger.error("Invalid input. Please enter numbers separated by commas.")
            return
    
    if not tickers_to_delete:
        logger.info("No tickers selected for deletion.")
        return
    
    # Confirm deletion
    print(f"\nYou are about to delete records for: {', '.join(tickers_to_delete)}")
    confirm = input("Are you sure? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        logger.info("Deletion cancelled by user.")
        return
    
    # Delete records
    total_deleted = 0
    for ticker in tickers_to_delete:
        deleted = delete_ticker_records(ticker)
        total_deleted += deleted
    
    logger.info(f"Deletion complete. Total records deleted: {total_deleted}")
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Total execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()

#TO RUN LOCALLY
# dotenv run -- .\.venv\Scripts\python.exe .\execution\data_fetch\delete_data_dynamo.py
