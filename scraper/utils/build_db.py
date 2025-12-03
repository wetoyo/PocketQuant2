# build_db.py
from pathlib import Path
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


def write_data_to_db(data_dict, db_path="database.db"):
    """
    Write a dict of DataFrames (raw data or features) to SQLite DB.
    Each ticker gets its own table: <TICKER>.
    Updates existing rows based on DATE (PRIMARY KEY).
    """
    if not isinstance(data_dict, dict):
        raise ValueError("data_dict must be a dict of DataFrames keyed by ticker")

    conn = create_or_connect_db(db_path)
    cursor = conn.cursor()

    for ticker, df in data_dict.items():
        # Use ticker as table name for raw data, or whatever key is passed
        table_name = ticker.upper()

        if "DATE" not in df.columns:
            # Try to see if index is datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "DATE"})
            
            if "DATE" not in df.columns:
                 raise KeyError(f"Missing 'DATE' column in ticker {ticker} DataFrame")
        
        df = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d %H:%M:%S")

        columns_with_types = []
        for col in df.columns:
            if df[col].dtype == object:
                columns_with_types.append(f'"{col}" TEXT')
            else:
                columns_with_types.append(f'"{col}" REAL')
        columns_with_types_sql = ", ".join(columns_with_types)

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            {columns_with_types_sql},
            PRIMARY KEY(DATE)
        )
        """
        cursor.execute(create_sql)
        
        placeholders = ", ".join(["?"] * len(df.columns))
        insert_sql = f"""
        INSERT OR REPLACE INTO "{table_name}" ({', '.join(df.columns)})
        VALUES ({placeholders})
        """
        cursor.executemany(insert_sql, df.values.tolist())
        conn.commit()
        print(f"Data written/updated in table '{table_name}'")

    conn.close()


def get_ticker_date_range(db_path, ticker):
    """
    Get the min and max date for a ticker in the DB.
    Returns (min_date, max_date) as strings, or (None, None) if table doesn't exist.
    """
    table_name = ticker.upper()
    conn = create_or_connect_db(db_path)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    if cursor.fetchone() is None:
        conn.close()
        return None, None

    cursor.execute(f'SELECT MIN(DATE), MAX(DATE) FROM "{table_name}"')
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0], result[1]
    return None, None


def read_table(db_path, table_name):
    """Read an entire table into a pandas DataFrame."""
    conn = create_or_connect_db(db_path)
    try:
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
    except Exception as e:
        print(f"Error reading table {table_name}: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    
    if not df.empty and "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
        
    return df


def read_by_ticker(db_path, ticker):
    """Read a single ticker's table."""
    table_name = ticker.upper()
    return read_table(db_path, table_name)


def read_by_date(db_path, ticker, start_date=None, end_date=None):
    """Return rows filtered by date from a ticker's table."""
    table_name = ticker.upper()
    conn = create_or_connect_db(db_path)
    
    # Check if table exists first to avoid error
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    if cursor.fetchone() is None:
        conn.close()
        return pd.DataFrame()

    query = f'SELECT * FROM "{table_name}" WHERE 1=1'
    params = []

    if start_date:
        query += " AND DATE >= ?"
        params.append(str(start_date))
    if end_date:
        query += " AND DATE <= ?"
        params.append(str(end_date))

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    
    if not df.empty and "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])
        
    return df


def write_options_to_db(options_data, db_name="options_data.db"):
    """
    Write options data to SQLite DB.
    options_data structure: { ticker: { expiration: { 'calls': df, 'puts': df } } }
    Creates one table per ticker containing all expirations and types.
    """
    if not options_data:
        print("No options data to write.")
        return

    conn = create_or_connect_db(db_name)
    
    for ticker, dates_dict in options_data.items():
        all_dfs = []
        for date, chains in dates_dict.items():
            for kind, df in chains.items():
                if df is None or df.empty:
                    continue
                
                # Create a copy to avoid modifying original
                df_copy = df.copy()
                
                # Add metadata columns
                df_copy['EXPIRATION'] = date
                df_copy['OPTION_TYPE'] = kind
                df_copy['FETCH_DATE'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Ensure columns are strings to avoid DB issues
                df_copy.columns = [str(c) for c in df_copy.columns]
                
                all_dfs.append(df_copy)
        
        if not all_dfs:
            print(f"No options data found for {ticker}")
            continue

        combined_df = pd.concat(all_dfs, ignore_index=True)
        table_name = ticker.upper()
        
        try:
            # Append new data
            combined_df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"Options data appended to table '{table_name}' in {db_name}")
            
            # Create index on FETCH_DATE if it doesn't exist
            cursor = conn.cursor()
            index_name = f"idx_{table_name}_fetch_date"
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (FETCH_DATE)")
            conn.commit()
            
        except Exception as e:
            print(f"Error writing options for {ticker}: {e}")

    conn.close()
