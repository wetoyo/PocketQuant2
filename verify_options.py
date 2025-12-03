import sys
from pathlib import Path
import logging

# Add project root to path
root = Path(__file__).parent
sys.path.append(str(root))

from scraper.setup import BaseSetup
from configs.paths import DATA_RAW

def verify_options():
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
    
    # Check if options files exist
    options_folder = DATA_RAW / "options" / "AAPL"
    if options_folder.exists() and any(options_folder.iterdir()):
        print(f"SUCCESS: Options data found in {options_folder}")
        for file in options_folder.iterdir():
            print(f"  - {file.name}")
    else:
        print(f"FAILURE: No options data found in {options_folder}")

if __name__ == "__main__":
    verify_options()
