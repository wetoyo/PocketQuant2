import sys
from pathlib import Path
import logging
import sqlite3
import pandas as pd

# Add project root to path
root = Path(__file__).parent
sys.path.append(str(root))

from scraper.setup import BaseSetup
from configs.paths import DATABASE_PATH

def verify_options_db():
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing BaseSetup with include_options=True...")
    setup = BaseSetup(
        tickers=["AAPL"],
        start_date="2023-10-01",
        end_date="2023-10-05",
        include_options=True
    )
    
    print("Running pipeline...")
    setup.run_pipeline()
    
    # Check if options data exists in DB
    db_path = DATABASE_PATH / "options_data.db"
    if not db_path.exists():
        print(f"FAILURE: DB file not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    try:
        # Check total rows
        count_query = "SELECT COUNT(*) FROM AAPL"
        count = pd.read_sql(count_query, conn).iloc[0, 0]
        print(f"Total rows in AAPL table: {count}")
        
        # Check unique fetch dates
        fetch_query = "SELECT DISTINCT FETCH_DATE FROM AAPL"
        fetch_dates = pd.read_sql(fetch_query, conn)
        print("Unique FETCH_DATEs:", fetch_dates['FETCH_DATE'].tolist())
        
        # Check expiration range
        exp_query = "SELECT MIN(EXPIRATION), MAX(EXPIRATION) FROM AAPL"
        exp_range = pd.read_sql(exp_query, conn).iloc[0]
        print(f"Expiration range: {exp_range[0]} to {exp_range[1]}")
        
        # Show sample data
        query = "SELECT * FROM AAPL LIMIT 5"
        df = pd.read_sql(query, conn)
        print("Sample data:")
        print(df.head())
            
    except Exception as e:
        print(f"FAILURE: Error querying DB: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    verify_options_db()
