# Backtest Examples (Minimal & Advanced)

This document contains copy-pasteable templates for building and running backtests on the PocketQuant2 framework.

---

## 1. Minimal Backtest: Simple Moving Average Crossover

This example demonstrates how to fetch stock price data, generate entry/exit signals from Moving Average indicators, and execute a signal-driven backtest.

```python
import pandas as pd
from backtester.backtester import Backtester

def main():
    # 1. Initialize Backtester
    # We specify tickers, start/end dates, and the technical indicators we want.
    bt = Backtester(
        tickers=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        scraper_type="yfinance",
        features_options={
            "ma_windows": [10, 30]  # Calculates MA_10 and MA_30
        }
    )

    # 2. Run Ingestion and Feature Extraction
    # This downloads the data (or loads from SQLite cache) and calculates indicators.
    bt.run_pipeline(save_folder_raw=None, save_folder_features=None)

    # 3. Retrieve Price and Features Data
    prices = bt.get_price_data(price_col="CLOSE")
    features = bt.get_ticker_features("AAPL")

    # 4. Generate Crossover Signals
    # Ensure index is datetime-aligned
    if "DATE" in features.columns:
        features = features.set_index(pd.to_datetime(features["DATE"]))

    ma_10 = features["MA_10"]
    ma_30 = features["MA_30"]

    # Entry: Fast MA crosses above Slow MA
    entries = (ma_10 > ma_30) & (ma_10.shift(1) <= ma_30.shift(1))
    
    # Exit: Fast MA crosses below Slow MA
    exits = (ma_10 < ma_30) & (ma_10.shift(1) >= ma_30.shift(1))

    # 5. Format Signals for Multi-ticker Support
    entries_df = pd.DataFrame({"AAPL": entries})
    exits_df = pd.DataFrame({"AAPL": exits})

    # 6. Execute Backtest
    portfolio = bt.run_backtest(entries_df, exits_df, freq="D", init_cash=10000)

    # 7. Print Performance Stats
    print("\n=== Performance Stats ===")
    print(bt.get_stats())

if __name__ == "__main__":
    main()
```

---

## 2. Advanced Backtest: Machine Learning (XGBoost) Strategy

This script demonstrates how to train an XGBoost classifier to predict whether the next period's price is higher, apply strict feature shifting to prevent lookahead leakage, and execute a backtest using model predictions.

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from backtester.backtester import Backtester

def main():
    ticker = "NVDA"
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    # 1. Initialize data and feature pipeline
    bt = Backtester(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        features_options={
            "returns": True,
            "ma_windows": [5, 20],
            "rsi_windows": [14]
        }
    )
    bt.run_pipeline(save_folder_raw=None, save_folder_features=None)

    # 2. Retrieve data
    raw_df = bt.get_ticker_data(ticker)
    features_df = bt.get_ticker_features(ticker)

    # Convert date column to index for joining
    raw_df["DATE"] = pd.to_datetime(raw_df["DATE"])
    features_df["DATE"] = pd.to_datetime(features_df["DATE"])
    
    df = pd.merge(raw_df, features_df, on="DATE", how="inner").set_index("DATE")

    # 3. Create target and shift features to prevent leakage
    df["target"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)
    df = df.dropna()

    feature_cols = ["RETURN", "MA_5", "MA_20", "RSI_14"]
    
    # Strict 1-bar shift to prevent lookahead bias
    X = df[feature_cols].shift(1)
    y = df["target"]

    # Drop NaN created by shift
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    # Chronological Split (80% Train, 20% Out-of-Sample Backtest)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4. Train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. Generate Trading Signals on Test Set
    # Predict direction (1 for up, 0 for down)
    predictions = model.predict(X_test)
    
    # Go long when model predicts upward movement
    entries = pd.Series(predictions == 1, index=X_test.index)
    
    # Exit long when model predicts downward or flat movement
    exits = pd.Series(predictions == 0, index=X_test.index)

    # Align signal index with prices
    entries_df = pd.DataFrame({ticker: entries})
    exits_df = pd.DataFrame({ticker: exits})

    # 6. Run Out-of-Sample Backtest
    portfolio = bt.run_backtest(entries_df, exits_df, freq="D", init_cash=10000)

    print("\n=== XGBoost Strategy Results ===")
    print(f"Total Return: {portfolio.total_return().values[0]:.2%}")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio().values[0]:.2f}")

if __name__ == "__main__":
    main()
```

---

## 3. Advanced Backtest: Event-Driven Reinvestment Portfolio

This example mimics the dividend tail hedging strategy in [expirments/svol_vixy_tailhedge.py](file:///d:/Files/Code/PocketQuant2/expirments/svol_vixy_tailhedge.py) where custom order volumes (shares) are calculated at specific dividend payout dates.

```python
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction
from backtester.backtester import Backtester

def run_event_backtest():
    start_date = "2021-05-12"
    initial_cash = 10000.0

    # 1. Fetch multi-ticker price & dividend data
    bt = Backtester(
        tickers=["SVOL", "VIXY"],
        start_date=start_date,
        features_options={"ma_windows": [20]}
    )
    bt.run_pipeline()

    # Get dividends and price matrices
    divs_dict = bt.get_dividends()
    svol_divs = divs_dict["SVOL"]
    prices = bt.get_price_data()

    # 2. Initialize Order Size DataFrame (zeros)
    order_sizes = pd.DataFrame(0.0, index=prices.index, columns=["SVOL", "VIXY"])

    # 3. Simulate holdings and dividend reinvestments chronologically
    first_date = prices.index[0]
    svol_shares = initial_cash / prices.loc[first_date, "SVOL"]
    vixy_shares = 0.0

    # Execute Initial Buy
    order_sizes.loc[first_date, "SVOL"] = svol_shares

    # Reinvest cash on each ex-dividend date
    for dt in svol_divs.index:
        # Align date if ex-dividend date falls on weekend/holiday
        if dt not in prices.index:
            dt = prices.index[prices.index.get_loc(dt, method='bfill')]

        # Calculate dividend cash received from current shares
        payout = svol_shares * svol_divs.loc[dt]
        if payout <= 0:
            continue

        # Reinvest rules: Check if price is above MA_20
        features = bt.get_ticker_features("SVOL")
        # Align index
        features.index = pd.to_datetime(features.index)
        
        is_uptrend = prices.loc[dt, "SVOL"] > features.loc[dt, "MA_20"]

        if is_uptrend:
            # Reinvest in SVOL
            buy_shares = payout / prices.loc[dt, "SVOL"]
            svol_shares += buy_shares
            order_sizes.loc[dt, "SVOL"] += buy_shares
        else:
            # Buy VIXY as a hedge
            buy_shares = payout / prices.loc[dt, "VIXY"]
            vixy_shares += buy_shares
            order_sizes.loc[dt, "VIXY"] += buy_shares

    # 4. Run vectorbt Backtest using orders
    portfolio = vbt.Portfolio.from_orders(
        close=prices,
        size=order_sizes,
        size_type=SizeType.Amount,
        direction=Direction.LongOnly,
        init_cash=initial_cash,
        freq="D"
    )

    print("\n=== Event-Driven Results ===")
    print(portfolio.stats())

if __name__ == "__main__":
    run_event_backtest()
```
