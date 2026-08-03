import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent
sys.path.append(str(root))

from scraper.api_clients.AlphaVantage import StockScraperAV

class TestAlphaVantage(unittest.TestCase):
    def test_extended_hours_param(self):
        """Test that extended_hours parameter is correctly passed to the API."""
        api_key = "TEST_KEY"
        scraper = StockScraperAV(
            api_key=api_key,
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-02",
            interval="5min",
            extended_hours=True
        )

        with patch('requests.get') as mock_get:
            # Mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Time Series (5min)": {
                    "2023-01-01 04:00:00": {
                        "1. open": "100.0",
                        "2. high": "101.0",
                        "3. low": "99.0",
                        "4. close": "100.5",
                        "5. volume": "1000"
                    }
                }
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            scraper.fetch_data()

            # Check if called with correct params
            args, kwargs = mock_get.call_args
            params = kwargs['params']
            
            self.assertEqual(params['function'], 'TIME_SERIES_INTRADAY')
            self.assertEqual(params['interval'], '5min')
            self.assertEqual(params['extended_hours'], 'true')
            print("Verified: extended_hours='true' param passed to request.")

    def test_no_extended_hours_param(self):
        """Test that extended_hours parameter is NOT passed when False."""
        api_key = "TEST_KEY"
        scraper = StockScraperAV(
            api_key=api_key,
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-02",
            interval="5min",
            extended_hours=False
        )

        with patch('requests.get') as mock_get:
            # Mock response (same as above)
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "Time Series (5min)": {
                    "2023-01-01 09:30:00": {
                        "1. open": "100.0",
                        "2. high": "101.0",
                        "3. low": "99.0",
                        "4. close": "100.5",
                        "5. volume": "1000"
                    }
                }
            }
            mock_get.return_value = mock_response

            scraper.fetch_data()

            # Check params
            args, kwargs = mock_get.call_args
            params = kwargs['params']
            
            self.assertNotIn('extended_hours', params)
            print("Verified: extended_hours param NOT passed when False.")

if __name__ == '__main__':
    unittest.main()
