# Setup: Initialize, Fetch, Retrieve

## Step 1 — Initialize the Backtester

```python
from backtester.backtester import Backtester

bt = Backtester(
    tickers=["AAPL"],               # Always UPPERCASE (gotchas.md §4)
    start_date="2022-01-01",
    end_date="2023-12-31",
    scraper_type="yfinance",        # "yfinance" | "alphavantage"
    interval="1d",                  # Match your execution window
    adjusted=True,                  # Use split/dividend-adjusted prices (rules.md §1)
    features_options={
        "ma_windows": [10, 30],
        "rsi_windows": [14],
        "returns": True,
    }
)
```

**Gotcha**: Features are computed with a built-in 1-bar forward shift (gotchas.md §2).
The feature labelled `MA_10` at date `t` reflects the MA computed at `t-1`. This is intentional — do not shift again downstream.

---

## Step 2 — Fetch Data and Build Features

```python
bt.run_pipeline(save_folder_raw=None, save_folder_features=None)
```

`run_pipeline()` checks the SQLite cache before hitting the API. It only makes a network request if the requested date range is not already in `market_data.db` (data_pipeline.md §4).

After this call:
- `bt.data["AAPL"]`     → raw OHLCV DataFrame
- `bt.features["AAPL"]` → pre-shifted feature DataFrame

---

## Step 3 — Retrieve Price and Feature Data

```python
prices   = bt.get_price_data(price_col="CLOSE")   # DatetimeIndex × [AAPL]
features = bt.get_ticker_features("AAPL")          # DatetimeIndex × [MA_10, MA_30, ...]
```

If the index still has a `DATE` column rather than a DatetimeIndex:
```python
features = features.set_index(pd.to_datetime(features["DATE"]))
```
