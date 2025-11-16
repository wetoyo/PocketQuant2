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


def write_features_dict_to_db(features_dict, db_path="database.db"):
    """
    Write a dict of features DataFrames to SQLite DB.
    Each ticker gets its own table: <TICKER>_features.
    Updates existing rows based on DATE (PRIMARY KEY).
    """
    if not isinstance(features_dict, dict):
        raise ValueError("features_dict must be a dict of DataFrames keyed by ticker")

    conn = create_or_connect_db(db_path)
    cursor = conn.cursor()

    for ticker, df in features_dict.items():
        table_name = f"{ticker.upper()}_features"

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
        print(f"Features written/updated in table '{table_name}'")

    conn.close()


def read_table(db_path, table_name):
    """Read an entire table into a pandas DataFrame."""
    conn = create_or_connect_db(db_path)
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
    conn.close()
    return df


def read_by_ticker(db_path, ticker):
    """Read a single ticker's features table."""
    table_name = f"{ticker.upper()}_features"
    return read_table(db_path, table_name)


def read_by_date(db_path, ticker, start_date=None, end_date=None):
    """Return rows filtered by date from a ticker's features table."""
    table_name = f"{ticker.upper()}_features"
    conn = create_or_connect_db(db_path)
    query = f'SELECT * FROM "{table_name}" WHERE 1=1'
    params = []

    if start_date:
        query += " AND DATE >= ?"
        params.append(start_date)
    if end_date:
        query += " AND DATE <= ?"
        params.append(end_date)

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df


def read_multiple_tickers(db_path, tickers):
    """
    Read multiple tickers' features and combine into one DataFrame.
    Adds TICKER column if missing.
    """
    dfs = []
    for ticker in tickers:
        df = read_by_ticker(db_path, ticker)
        if "TICKER" not in df.columns:
            df["TICKER"] = ticker.upper()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def csvs_to_db(folder_path, db_path="database.db"):
    """
    Read all CSV files in a folder and write them as ticker tables in the DB.
    Each CSV must have a 'DATE' column.
    Existing rows are updated (INSERT OR REPLACE).
    """
    folder_path = Path(folder_path)
    for csv_file in folder_path.glob("*.csv"):
        ticker = csv_file.stem.split("_")[0].upper()  # assumes CSV naming like AAPL_features.csv
        df = pd.read_csv(csv_file)
        write_features_dict_to_db({ticker: df}, db_path=db_path)
        print(f"Written CSV {csv_file} to table {ticker}_features")
