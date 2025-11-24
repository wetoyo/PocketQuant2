import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import shutil
import tempfile

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scraper.api_clients.YFinance import StockScraper
from scraper.setup import BaseSetup
from scraper.utils.feature_builder import build_features
from scraper.utils.build_db import create_or_connect_db, write_features_dict_to_db, read_table

class TestPocketQuant(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.test_data_raw = Path(self.test_dir) / "data" / "raw"
        self.test_data_features = Path(self.test_dir) / "data" / "features"
        self.test_db_path = Path(self.test_dir) / "test.db"
        
        self.test_data_raw.mkdir(parents=True, exist_ok=True)
        self.test_data_features.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    @patch('yfinance.download')
    def test_stock_scraper_yfinance(self, mock_yf_download):
        # Mock yfinance data
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        mock_df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Adj Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)
        mock_df.index.name = "Date"
        mock_yf_download.return_value = mock_df

        scraper = StockScraper(
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-05",
            interval="1d"
        )
        
        # Test fetch_data
        scraper.fetch_data()
        self.assertIn("AAPL", scraper.data)
        self.assertFalse(scraper.data["AAPL"].empty)
        self.assertIn("CLOSE", scraper.data["AAPL"].columns)

        # Test clean_data
        scraper.clean_data()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(scraper.data["AAPL"]["DATE"]))

        # Test save_data
        scraper.save_data(folder=str(self.test_data_raw), format="csv")
        self.assertTrue((self.test_data_raw / "AAPL.csv").exists())

    def test_feature_builder(self):
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'DATE': dates,
            'OPEN': np.random.rand(50) * 100,
            'HIGH': np.random.rand(50) * 100,
            'LOW': np.random.rand(50) * 100,
            'CLOSE': np.random.rand(50) * 100,
            'VOLUME': np.random.randint(100, 1000, 50)
        })
        
        # Test build_features
        features_df = build_features(
            df=df,
            save_path=self.test_data_features,
            ma_windows=[5, 10],
            bb_windows=[20],
            rsi_windows=[14]
        )
        
        self.assertFalse(features_df.empty)
        self.assertIn("MA_5", features_df.columns)
        self.assertIn("RSI_14", features_df.columns)
        self.assertTrue((self.test_data_features / "features.csv").exists())

    def test_database_operations(self):
        # Create sample features data
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'DATE': dates,
            'CLOSE': [100, 101, 102, 103, 104],
            'MA_5': [100, 100.5, 101, 101.5, 102]
        })
        
        features_dict = {"AAPL": df}
        
        # Test write_features_dict_to_db
        write_features_dict_to_db(features_dict, db_path=self.test_db_path)
        self.assertTrue(self.test_db_path.exists())
        
        # Test read_table
        read_df = read_table(self.test_db_path, "AAPL_features")
        self.assertFalse(read_df.empty)
        self.assertEqual(len(read_df), 5)

    def test_base_setup_pipeline(self):
        # Mock StockScraper behavior inside BaseSetup
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        mock_df = pd.DataFrame({
            'DATE': dates,
            'OPEN': np.random.rand(50) * 100,
            'HIGH': np.random.rand(50) * 100,
            'LOW': np.random.rand(50) * 100,
            'CLOSE': np.random.rand(50) * 100,
            'VOLUME': np.random.randint(100, 1000, 50)
        })
        
        # We need to mock the scraper instance that BaseSetup creates
        with patch('scraper.api_clients.YFinance.StockScraper') as MockScraper:
            instance = MockScraper.return_value
            instance.data = {"AAPL": mock_df}
            instance.fetch_data.return_value = None
            instance.clean_data.return_value = None
            instance.save_data.return_value = None

            setup = BaseSetup(
                tickers=["AAPL"],
                start_date="2023-01-01",
                end_date="2023-02-01",
                features_options={"ma_windows": [5]}
            )
            
            # Override db_path to use temp dir
            setup.db_path = self.test_db_path

            # Run pipeline
            features = setup.run_pipeline(
                save_folder_raw=self.test_data_raw,
                save_folder_features=self.test_data_features,
                db_path=self.test_db_path
            )
            
            self.assertIn("AAPL", features)
            self.assertIn("MA_5", features["AAPL"].columns)
            
            # Verify DB was written
            read_df = read_table(self.test_db_path, "AAPL_features")
            self.assertFalse(read_df.empty)

if __name__ == '__main__':
    unittest.main()
