import sys
from pathlib import Path

# Add project root to path for imports
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from research.evaluate_features import run_evaluation_system

def result_features_test():
    """
    Test the feature evaluation system with specific feature options.
    """
    custom_options = {
        "returns": True,
        "log_returns": True,
        "ma_windows": [10, 50, 200],  # Different MA windows
        "bb_windows": [20],
        "rsi_windows": [14, 21],      # Added another RSI window
        "macd_params": [(12, 26, 9)],
        "vol_windows": [20],          # Changed volatility window
        "atr_windows": [14]
    }

    print("Running feature evaluation with custom options...")
    run_evaluation_system(
        tickers=["AAPL"], 
        start_date="2025-01-01", 
        interval="1m", 
        feature_options=custom_options
    )

if __name__ == "__main__":
    result_features_test()
