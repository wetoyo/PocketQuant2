import sys
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbt as vbt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.backtester import Backtester

def test_backtester_simple():
    print("Testing Backtester class...")
    
    # Setup dummy data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    price_data = pd.DataFrame({
        "CLOSE": np.random.randn(100).cumsum() + 100
    }, index=dates)
    # Ensure DATE column exists as per BaseSetup/Backtester expectations if it relies on it
    # The get_price_data method checks for DATE column or index.
    price_data["DATE"] = price_data.index 
    
    # Initialize Backtester
    # We pass empty tickers list as we manually inject data
    bt = Backtester(tickers=["TEST"])
    bt.data = {"TEST": price_data} # Inject data manually to avoid fetching
    
    # Test get_price_data
    prices = bt.get_price_data()
    print(f"Prices shape: {prices.shape}")
    if prices.empty:
        raise ValueError("Prices DataFrame is empty")
    
    # Create signals using vectorbt
    fast_ma = vbt.MA.run(prices, 10)
    slow_ma = vbt.MA.run(prices, 20)
    
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    # Run backtest
    print("Running backtest...")
    portfolio = bt.run_backtest(entries, exits, freq="D")
    
    print("Portfolio created successfully.")
    print(f"Total Return: {portfolio.total_return().values[0]}")
    
    # Check stats
    stats = bt.get_stats()
    print("Stats retrieved.")
    
    return True

if __name__ == "__main__":
    try:
        test_backtester_simple()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
