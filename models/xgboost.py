import sys
from pathlib import Path
root = Path(__file__).parent.parent  # project_root
sys.path.append(str(root))
from scraper.setup import BaseSetup

test = BaseSetup(
    tickers=["AAPL","MSFT"],
    scraper_type="yfinance",
    start_date="2020-01-01",
    end_date="2023-01-01",
    interval="1d",
    adjusted=True,
    fill_missing=True,
)
test.run_pipeline()
