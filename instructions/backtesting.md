# Creating and Running Backtests

This document provides step-by-step instructions for implementing, running, and debugging backtests within the PocketQuant2 framework.

---

## 1. Core Interfaces

All backtests rely on the `Backtester` class in [backtester/backtester.py](file:///d:/Files/Code/PocketQuant2/backtester/backtester.py). It provides two methods for portfolio execution:

### A. Signal-Driven execution (`run_backtest`)
Best for traditional strategies where you generate boolean entry (buy/long) and exit (sell/flat) signals. Under the hood, it uses:
```python
vbt.Portfolio.from_signals(prices, entries, exits, **kwargs)
```

### B. Event-Driven execution (`from_orders`)
Best for complex strategies (e.g., dividend reinvestment, multi-asset routing) where you calculate exact trade quantities (shares) on specific events. Under the hood, it uses:
```python
vbt.Portfolio.from_orders(close, size, size_type, direction, **kwargs)
```

---

## 2. Signal Alignment and Date Mismatch

Before executing a backtest, the `Backtester` aligns price data and trading signals in time. This is a critical step because indicators and price series might have different date ranges or missing bars.

1.  **Reshaping Prices**: `get_price_data(price_col)` aggregates prices from multiple tickers into a single wide DataFrame where columns are ticker symbols and the index is a timezone-naive `DatetimeIndex`.
2.  **Signal Alignment**: In `run_backtest`, the overlapping indices are extracted using:
    ```python
    common_index = prices.index.intersection(entries.index)
    ```
    If `common_index` is empty, it raises:
    ```
    ValueError: No overlapping dates between prices and signals.
    ```
3.  **Slicing**: Prices, entries, and exits are sliced to match the `common_index` before being passed to vectorbt.

---

## 3. Step-by-Step Backtest Implementation (Signal-Driven)

Follow these steps to create a signal-driven backtest:

### Step 1: Initialize the Backtester
Specify the tickers, date range, interval, and technical feature windows:
```python
from backtester.backtester import Backtester

features_options = {
    "ma_windows": [10, 30],
    "returns": True
}

bt = Backtester(
    tickers=["AAPL"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    scraper_type="yfinance",
    features_options=features_options
)
```

### Step 2: Fetch Data and Features
Run the data pipeline to download data (or load from SQLite cache) and calculate indicators:
```python
bt.run_pipeline(save_folder_raw=None, save_folder_features=None)
```

### Step 3: Get Price and Feature Data
Retrieve the price series and calculated indicators:
```python
prices = bt.get_price_data(price_col="CLOSE")
features = bt.get_ticker_features("AAPL")
```

### Step 4: Calculate Entry and Exit Signals
Generate boolean signals based on indicator crossovers:
```python
ma_10 = features["MA_10"]
ma_30 = features["MA_30"]

# Long crossover entry
entries = (ma_10 > ma_30) & (ma_10.shift(1) <= ma_30.shift(1))

# Long exit
exits = (ma_10 < ma_30) & (ma_10.shift(1) >= ma_30.shift(1))
```

### Step 5: Wrap Signals into DataFrames
Vectorbt requires signal structures to match the columns of price data:
```python
import pandas as pd
entries_df = pd.DataFrame({"AAPL": entries})
exits_df = pd.DataFrame({"AAPL": exits})
```

### Step 6: Execute and Analyze
Run the backtest and extract performance statistics:
```python
portfolio = bt.run_backtest(entries_df, exits_df, freq="D", init_cash=10000)
stats = bt.get_stats()
print(stats)
```

---

## 4. Step-by-Step Backtest Implementation (Event-Driven)

Refer to the historical example in [expirments/svol_vixy_tailhedge.py](file:///d:/Files/Code/PocketQuant2/expirments/svol_vixy_tailhedge.py) for implementing an event-driven backtest:

1.  **Initialize Backtester**: Download prices and dividend history using `bt.get_dividends()`.
2.  **Create Order Sizing DataFrame**: Initialize a DataFrame of zeros matching the shape of prices:
    ```python
    order_size = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    ```
3.  **Simulate Decisions**: Loop through event dates (e.g., dividend ex-dates) and compute target share numbers:
    ```python
    for dt in div_dates:
        # Check market data at event timestamp 'dt'
        # Assign shares to buy/sell to order_size.loc[dt, ticker]
    ```
4.  **Execute via `from_orders`**:
    ```python
    import vectorbt as vbt
    from vectorbt.portfolio.enums import SizeType, Direction

    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=order_size,
        size_type=SizeType.Amount,         # Amount of shares
        direction=Direction.LongOnly,      # Long only
        init_cash=10000.0,
        freq="D"
    )
    ```
