# build_db.py
from pathlib import Path
import os
import sys
import sqlite3
import pandas as pd
root = Path(__file__).parent.parent.parent  # project_root
sys.path.append(str(root))
from configs.paths import DATABASE_PATH

def create_or_connect_db(db_path="database.db"):
    """Create a new SQLite database or connect to an existing one."""
    db_file = DATABASE_PATH / db_path
    conn = sqlite3.connect(str(db_file))
    return conn

def csvs_to_db(folder_path, db_path="database.db"):
    """
    Read all CSV files in a folder and write them to a SQL database.
    
    Each CSV will become a table with the same name as the file (without .csv)
    Existing tables are replaced.
    """
    folder_path = Path(folder_path)
    db_file = DATABASE_PATH / db_path
    conn = sqlite3.connect(str(db_file))
    
    for csv_file in folder_path.glob("*.csv"):
        table_name = csv_file.stem  # filename without suffix
        df = pd.read_csv(csv_file)
        
        # Write to DB (replace if table exists)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Written {csv_file} -> table '{table_name}'")
    
    conn.close()
    print(f"All CSVs from {folder_path} written to {db_path}")


def append_csvs_to_db(csv_files, db_path="database.db"):
    """
    Read specific CSV files and append them to their respective tables in the DB.
    
    csv_files: list of paths to CSV files
    Tables are named after the CSV filenames (without .csv)
    Existing rows are kept; new rows are appended.
    """
    db_file = DATABASE_PATH / db_path
    conn = sqlite3.connect(str(db_file))
    
    for file in csv_files:
        csv_file = Path(file)
        if not csv_file.exists():
            print(f"File not found: {csv_file}")
            continue
        
        table_name = csv_file.stem
        df = pd.read_csv(csv_file)
        
        # Append to DB (create table if it doesn't exist)
        df.to_sql(table_name, conn, if_exists='append', index=False)
        print(f"Appended {csv_file} -> table '{table_name}'")
    
    conn.close()
    print(f"All specified CSVs appended to {db_path}")

def read_table(db_path, table_name):
    """Read an entire table into a pandas DataFrame."""
    db_file = DATABASE_PATH / db_path
    conn = sqlite3.connect(str(db_file))
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def read_by_ticker(db_path, table_name, ticker):
    """Return rows for a single ticker."""
    db_file = DATABASE_PATH / db_path
    conn = sqlite3.connect(str(db_file))
    df = pd.read_sql(
        f"SELECT * FROM {table_name} WHERE ticker = ?", 
        conn, 
        params=(ticker,)
    )
    conn.close()
    return df

def read_by_tickers(db_path, table_name, tickers):
    """Return rows for a list of tickers."""
    db_file = DATABASE_PATH / db_path
    conn = sqlite3.connect(str(db_file))
    placeholders = ','.join('?' for _ in tickers)
    query = f"SELECT * FROM {table_name} WHERE ticker IN ({placeholders})"
    df = pd.read_sql(query, conn, params=tickers)
    conn.close()
    return df

def read_by_date(db_path, table_name, start_date=None, end_date=None):
    """
    Return rows filtered by date.
    Dates should be strings in 'YYYY-MM-DD' format.
    """
    db_file = DATABASE_PATH / db_path
    conn = sqlite3.connect(str(db_file))
    query = f"SELECT * FROM {table_name} WHERE 1=1"
    params = []
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df

# Example usage
if __name__ == "__main__":
    # Example: write all CSVs in 'data/' to DB
    csv_folder = "data"
    db_file = "Stock.db"
    csvs_to_db(csv_folder, db_file)
    
    # Example: read table
    df = read_table(db_file, "APPL")
    print(df.head())
    
    # Example: read by ticker
    df_aapl = read_by_ticker(db_file, "APPL", "AAPL")
    print(df_aapl.head())
    
    # Example: read by tickers
    df_group = read_by_tickers(db_file, "APPL_MSFT", ["AAPL", "MSFT"])
    print(df_group.head())
    
    # Example: read by date
    df_dates = read_by_date(db_file, "APPL", start_date="2025-01-01", end_date="2025-01-31")
    print(df_dates.head())
