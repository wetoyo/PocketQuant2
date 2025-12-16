import sys
from pathlib import Path
import pandas as pd
import vectorbt as vbt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backtester import Backtester

def run_backtest():
    
    features_options = {
        "returns": True,
        "log_returns": True,
        "ma_windows": [10, 30],
    }
    
    bt = Backtester(
        tickers=["NVDA"],
        start_date="2025-11-01",
        end_date="2025-11-30",
        scraper_type="yfinance",
        interval="5m",
        features_options=features_options
    )
    
    bt.run_pipeline()
    
    prices = bt.get_price_data(price_col="CLOSE")
    
    if prices.empty:
        print("No data found for NVDA. Aborting.")
        return

    print("Retrieving feature data...")
    features = bt.get_ticker_features("NVDA")
    
    if features is None or features.empty:
        print("Features not found.")
        return

    ma_10 = features["MA_10"]
    ma_30 = features["MA_30"]
    entries = (ma_10 > ma_30) & (ma_10.shift(1) <= ma_30.shift(1))
    exits = (ma_10 < ma_30) & (ma_10.shift(1) >= ma_30.shift(1))
    
    entries_df = pd.DataFrame({"NVDA": entries})
    exits_df = pd.DataFrame({"NVDA": exits})
    
    print("Running backtest...")
    portfolio = bt.run_backtest(entries_df, exits_df, freq="D", init_cash=10000)
    
    print("\n--- Backtest Results ---")
    print(f"Total Return: {portfolio.total_return().values[0]:.2%}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio().values[0]:.2f}")
    print(f"Max Drawdown: {portfolio.max_drawdown().values[0]:.2%}")
    print("------------------------")

if __name__ == "__main__":
    try:
        run_backtest()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
