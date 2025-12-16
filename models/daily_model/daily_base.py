import sys
from pathlib import Path
import pandas as pd
import vectorbt as vbt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backtester import Backtester

def run_backtest():
    
    ticker = "NVDA"
    benchmark_ticker = "QQQ"
    
    # Define separate periods to avoid look-ahead bias
    beta_start = "2025-10-20"  # Historical period for beta calculation
    beta_end = "2025-11-29"     # End of beta period
    backtest_start = "2025-12-01"  # Out-of-sample backtest period
    backtest_end = "2025-12-13"

    features_options = {
        "returns": True,
        "log_returns": True,
        "ma_windows": [10, 30],
    }
    
    # Fetch data for entire period (beta + backtest)
    bt = Backtester(
        tickers=[ticker, benchmark_ticker],
        start_date=beta_start,
        end_date=backtest_end,
        scraper_type="yfinance",
        interval="5m",
        features_options=features_options
    )
    
    bt.run_pipeline()
    
    prices = bt.get_price_data(price_col="CLOSE")
    
    if prices.empty:
        print(f"No data found for {ticker}. Aborting.")
        return

    print("Retrieving feature data...")
    features = bt.get_ticker_features(ticker)
    
    if features is None or features.empty:
        print("Features not found.")
        return

    # Filter features to backtest period only
    features_backtest = features[backtest_start:backtest_end]
    
    ma_10 = features_backtest["MA_10"]
    ma_30 = features_backtest["MA_30"]
    entries = (ma_10 > ma_30) & (ma_10.shift(1) <= ma_30.shift(1))
    exits = (ma_10 < ma_30) & (ma_10.shift(1) >= ma_30.shift(1))
    
    entries_df = pd.DataFrame({ticker: entries})
    exits_df = pd.DataFrame({ticker: exits})
    
    print("Running backtest...")
    portfolio = bt.run_backtest(entries_df, exits_df, freq="5min", init_cash=10000)
    
    print("\n--- Backtest Results ---")
    print(f"Total Return: {portfolio.total_return().values[0]:.2%}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio().values[0]:.2f}")
    print(f"Max Drawdown: {portfolio.max_drawdown().values[0]:.2%}")
    
    
    # Calculate Alpha and Beta using ONLY the historical period (before backtest)
    # Pass date range to ensure we only use pre-backtest data
    metrics = bt.calculate_alpha_beta(
        benchmark_ticker, 
        ticker, 
        freq="D",
        date_range=(beta_start, beta_end)  # Only use historical data
    )
    if metrics:
        print(f"Beta: {metrics['beta']:.2f}")
        print(f"Alpha (Annualized): {metrics['alpha']:.2%}")
        print(f"Bars Used: {metrics['n_bars_used']} (Required for 20% SE: {metrics['required_n']})")

    print("------------------------")

    print("------------------------")
    print("------------------------")

if __name__ == "__main__":
    try:
        run_backtest()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
