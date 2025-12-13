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


def write_data_to_db(data_dict, db_path="database.db", interval="1d"):
    """
    Write a dict of DataFrames (raw data or features) to SQLite DB.
    Each ticker gets its own table: <TICKER>.
    Updates existing rows based on composite key (DATE, INTERVAL).
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
        
        # Add INTERVAL column if not present
        if "INTERVAL" not in df.columns:
            df["INTERVAL"] = interval
        
        if pd.api.types.is_datetime64_any_dtype(df["DATE"]):
            df["DATE"] = df["DATE"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Build column definitions with types
        columns_with_types = []
        for col in df.columns:
            if col == "INTERVAL":
                columns_with_types.append(f'"{col}" TEXT')
            elif df[col].dtype == object:
                columns_with_types.append(f'"{col}" TEXT')
            else:
                columns_with_types.append(f'"{col}" REAL')
        columns_with_types_sql = ", ".join(columns_with_types)

        # Create table with composite primary key
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            {columns_with_types_sql},
            PRIMARY KEY(DATE, INTERVAL)
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
        print(f"Data written/updated in table '{table_name}' for interval '{interval}'")

    conn.close()


def get_ticker_date_range(db_path, ticker, interval="1d"):
    """
    Get the min and max date for a ticker in the DB for a specific interval.
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

    # Query for specific interval
    cursor.execute(f'SELECT MIN(DATE), MAX(DATE) FROM "{table_name}" WHERE INTERVAL = ?', (interval,))
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


def read_by_date(db_path, ticker, start_date=None, end_date=None, interval="1d"):
    """Return rows filtered by date and interval from a ticker's table."""
    table_name = ticker.upper()
    conn = create_or_connect_db(db_path)
    
    # Check if table exists first to avoid error
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    if cursor.fetchone() is None:
        conn.close()
        return pd.DataFrame()

    query = f'SELECT * FROM "{table_name}" WHERE INTERVAL = ?'
    params = [interval]

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
    Avoids rewriting existing entries by checking for duplicates.
    """
    if not options_data:
        print("No options data to write.")
        return

    conn = create_or_connect_db(db_name)
    cursor = conn.cursor()
    
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
                
                # Get current date and adjust to last weekday if needed
                current_date = pd.Timestamp.now().normalize()  # Remove time component
                weekday = current_date.weekday()
                
                # If Saturday (5) or Sunday (6), go back to Friday
                if weekday == 5:  # Saturday
                    current_date = current_date - pd.Timedelta(days=1)
                elif weekday == 6:  # Sunday
                    current_date = current_date - pd.Timedelta(days=2)
                
                df_copy['FETCH_DATE'] = current_date.strftime("%Y-%m-%d")
                
                # Ensure columns are strings to avoid DB issues
                df_copy.columns = [str(c) for c in df_copy.columns]
                
                all_dfs.append(df_copy)
        
        if not all_dfs:
            print(f"No options data found for {ticker}")
            continue

        combined_df = pd.concat(all_dfs, ignore_index=True)
        table_name = ticker.upper()
        
        try:
            # Check if table exists and read existing data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Read existing data
                existing_df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
                
                if not existing_df.empty:
                    # Identify key columns that define uniqueness
                    # Typically: strike, expiration, option_type, and potentially contractSymbol
                    key_columns = ['EXPIRATION', 'OPTION_TYPE']
                    
                    # Add strike if it exists
                    if 'strike' in combined_df.columns:
                        key_columns.append('strike')
                    
                    # Add contractSymbol if it exists (most unique identifier)
                    if 'contractSymbol' in combined_df.columns:
                        key_columns.append('contractSymbol')
                    
                    # Create a composite key for comparison
                    existing_df['_merge_key'] = existing_df[key_columns].astype(str).agg('||'.join, axis=1)
                    combined_df['_merge_key'] = combined_df[key_columns].astype(str).agg('||'.join, axis=1)
                    
                    # Filter out rows that already exist
                    new_rows = combined_df[~combined_df['_merge_key'].isin(existing_df['_merge_key'])]
                    new_rows = new_rows.drop(columns=['_merge_key'])
                    
                    if new_rows.empty:
                        print(f"No new options data to add for {ticker} (all entries already exist)")
                        continue
                    
                    print(f"Found {len(new_rows)} new entries out of {len(combined_df)} total for {ticker}")
                    combined_df = new_rows
                else:
                    print(f"Table '{table_name}' exists but is empty, writing all {len(combined_df)} rows")
            else:
                print(f"Creating new table '{table_name}' with {len(combined_df)} rows")
            
            # Append only new data
            combined_df.to_sql(table_name, conn, if_exists='append', index=False)
            print(f"Options data written to table '{table_name}' in {db_name}")
            
            # Create index on FETCH_DATE if it doesn't exist
            index_name = f"idx_{table_name}_fetch_date"
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (FETCH_DATE)")
            conn.commit()
            
        except Exception as e:
            print(f"Error writing options for {ticker}: {e}")

    conn.close()
