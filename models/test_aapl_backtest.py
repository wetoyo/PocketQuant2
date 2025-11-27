import sys
from pathlib import Path
import pandas as pd
import vectorbt as vbt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.backtester import Backtester

def run_aapl_backtest():
    print("Initializing Backtester for AAPL...")
    
    # Configure features we want
    features_options = {
        "returns": True,
        "log_returns": True,
        "ma_windows": [10, 30], # We want MA 10 and 30
    }
    
    bt = Backtester(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        scraper_type="yfinance",
        features_options=features_options
    )
    
    # Fetch data and build features
    print("Fetching/Loading data and building features...")
    bt.run_pipeline(save_folder_raw=None, save_folder_features=None)
    
    # Get price data
    prices = bt.get_price_data(price_col="CLOSE")
    print(f"Prices shape: {prices.shape}")
    
    if prices.empty:
        print("No data found for AAPL. Aborting.")
        return

    # Get Feature Data for AAPL
    print("Retrieving feature data...")
    features = bt.get_ticker_features("AAPL")
    
    if features is None or features.empty:
        print("Features not found.")
        return
        
    if "DATE" in features.columns:
        features = features.set_index(pd.to_datetime(features["DATE"]))

    ma_10 = features["MA_10"]
    ma_30 = features["MA_30"]

    # Define Strategy: Simple Moving Average Crossover
    print("Calculating signals using pre-built features...")
    
    # Create signals
    entries = (ma_10 > ma_30) & (ma_10.shift(1) <= ma_30.shift(1))
    exits = (ma_10 < ma_30) & (ma_10.shift(1) >= ma_30.shift(1))
    
    # Convert to DataFrame for vbt (even if single ticker, vbt likes aligned structures)
    # Or just pass Series if prices is Series.
    # But bt.run_backtest expects DataFrame inputs usually if prices is DataFrame.
    # Our get_price_data returns DataFrame.
    
    entries_df = pd.DataFrame({"AAPL": entries})
    exits_df = pd.DataFrame({"AAPL": exits})
    
    # Run Backtest
    print("Running backtest...")
    portfolio = bt.run_backtest(entries_df, exits_df, freq="D", init_cash=10000)
    
    # Print Stats
    print("\n--- Backtest Results ---")
    print(f"Total Return: {portfolio.total_return().values[0]:.2%}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio().values[0]:.2f}")
    print(f"Max Drawdown: {portfolio.max_drawdown().values[0]:.2%}")
    print("------------------------")

if __name__ == "__main__":
    try:
        run_aapl_backtest()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
