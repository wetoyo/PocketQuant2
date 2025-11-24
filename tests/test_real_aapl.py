
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scraper.setup import BaseSetup
from configs.paths import DATABASE_PATH, DATA_FEATURES

def test_aapl_fetch():
    print("Initializing BaseSetup for AAPL...")
    setup = BaseSetup(
        tickers=["AAPL"],
        db_name="market_data.db",
        start_date="2023-01-01",
        end_date="2023-12-31",
        interval="1d"
    )

    print("Running pipeline...")
    features = setup.run_pipeline()
    
    print("\nPipeline finished.")
    
    # Verify DB
    import sqlite3
    db_path = DATABASE_PATH / "market_data.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM AAPL")
        count = cursor.fetchone()[0]
        conn.close()
        print(f"Database 'market_data.db' contains {count} rows for AAPL.")
    else:
        print("ERROR: Database file not found!")

    # Verify CSV
    csv_path = DATA_FEATURES / "AAPL_features.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Feature CSV found at {csv_path}")
        print(f"CSV contains {len(df)} rows and {len(df.columns)} columns.")
        print("First 3 rows:")
        print(df.head(3))
    else:
        print(f"ERROR: Feature CSV not found at {csv_path}")

if __name__ == "__main__":
    test_aapl_fetch()
