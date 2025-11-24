
import sys
from pathlib import Path
import pandas as pd
import shutil
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scraper.setup import BaseSetup
from configs.paths import DATABASE_PATH, DATA_FEATURES

class TestScraperRefactor(unittest.TestCase):
    def setUp(self):
        self.test_ticker = "TEST_TICKER"
        self.db_name = "test_market_data.db"
        self.db_path = DATABASE_PATH / self.db_name
        self.features_path = DATA_FEATURES
        
        # Clean up before test
        if self.db_path.exists():
            self.db_path.unlink()
        
        # Clean up features
        feature_file = self.features_path / f"{self.test_ticker}_features.csv"
        if feature_file.exists():
            feature_file.unlink()

    def tearDown(self):
        # Clean up after test
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except PermissionError:
                pass # Might be locked
        
        feature_file = self.features_path / f"{self.test_ticker}_features.csv"
        if feature_file.exists():
            feature_file.unlink()

    @patch("scraper.setup.StockScraper")
    def test_pipeline_flow(self, MockScraper):
        # Setup mock
        mock_instance = MockScraper.return_value
        
        # Create dummy data
        dates = pd.date_range(start="2023-01-01", periods=5)
        dummy_df = pd.DataFrame({
            "DATE": dates,
            "OPEN": [100, 101, 102, 103, 104],
            "HIGH": [105, 106, 107, 108, 109],
            "LOW": [95, 96, 97, 98, 99],
            "CLOSE": [102, 103, 104, 105, 106],
            "VOLUME": [1000, 1100, 1200, 1300, 1400],
            "ADJ_CLOSE": [102, 103, 104, 105, 106]
        })
        # Scraper returns a dict of dfs
        mock_instance.data = {self.test_ticker: dummy_df}
        
        # Initialize BaseSetup
        setup = BaseSetup(
            tickers=[self.test_ticker],
            db_name=self.db_name,
            start_date="2023-01-01",
            end_date="2023-01-05"
        )
        
        # Run pipeline
        print("\nRunning pipeline first time (should fetch)...")
        setup.run_pipeline()
        
        # Verify fetch_data was called
        mock_instance.fetch_data.assert_called_once()
        
        # Verify DB created and has data
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(f"SELECT count(*) FROM {self.test_ticker}")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, 5, "Database should have 5 rows")
        
        # Verify CSV created
        csv_path = self.features_path / f"{self.test_ticker}_features.csv"
        self.assertTrue(csv_path.exists(), "Feature CSV should exist")
        
        # Run pipeline AGAIN (should NOT fetch)
        print("Running pipeline second time (should NOT fetch)...")
        
        # Reset mock to check calls
        mock_instance.fetch_data.reset_mock()
        
        setup.run_pipeline()
        
        # Verify fetch_data was NOT called
        mock_instance.fetch_data.assert_not_called()
        print("Verification successful: API not called on second run.")

if __name__ == "__main__":
    unittest.main()
