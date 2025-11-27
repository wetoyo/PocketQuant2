import sys
from pathlib import Path
import pandas as pd
import vectorbt as vbt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.backtester import Backtester

def run_multi_ticker_backtest():
    print("Initializing Backtester for AAPL and MSFT...")
    
    # Configure features
    features_options = {
        "returns": True,
        "ma_windows": [10, 30],
    }
    
    bt = Backtester(
        tickers=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        scraper_type="yfinance",
        features_options=features_options
    )
    
    # Fetch data and build features
    print("Fetching/Loading data and building features...")
    bt.run_pipeline(save_folder_raw=None, save_folder_features=None)
    
    # Get price data (still need combined for vbt execution, or we can build it from dict)
    prices = bt.get_price_data(price_col="CLOSE")
    print(f"Prices shape: {prices.shape}")
    
    if prices.empty:
        print("No data found. Aborting.")
        return

    # Calculate signals PER TICKER
    print("Calculating signals per ticker...")
    
    entries_dict = {}
    exits_dict = {}
    
    for ticker in ["AAPL", "MSFT"]:
        print(f"Processing {ticker}...")
        features = bt.get_ticker_features(ticker)
        
        if features is None or features.empty:
            print(f"  No features for {ticker}")
            continue
            
        # Ensure index is datetime for alignment
        if "DATE" in features.columns:
            features = features.set_index(pd.to_datetime(features["DATE"]))
            
        # Extract features
        ma_10 = features["MA_10"]
        ma_30 = features["MA_30"]
        
        # Calculate Logic
        entry_signal = (ma_10 > ma_30) & (ma_10.shift(1) <= ma_30.shift(1))
        exit_signal = (ma_10 < ma_30) & (ma_10.shift(1) >= ma_30.shift(1))
        
        entries_dict[ticker] = entry_signal
        exits_dict[ticker] = exit_signal
        
    # Align signals
    print("Aligning signals...")
    entries = bt.align_signals(entries_dict)
    exits = bt.align_signals(exits_dict)
    
    print(f"Entries shape: {entries.shape}")
    
    # Run Backtest
    print("Running backtest...")
    portfolio = bt.run_backtest(entries, exits, freq="D", init_cash=10000)
    
    # Print Stats
    print("\n--- Backtest Results ---")
    print(f"Total Return:\n{portfolio.total_return()}")
    print(f"\nSharpe Ratio:\n{portfolio.sharpe_ratio()}")
    print("------------------------")
    
    if len(portfolio.total_return()) == 2:
        print("SUCCESS: Results generated for both tickers using split logic.")
    else:
        print("FAILURE: Did not generate results for all tickers.")

if __name__ == "__main__":
    try:
        run_multi_ticker_backtest()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
