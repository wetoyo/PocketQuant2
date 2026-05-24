# Signals: Entry and Exit Generation

Signals must be **boolean DataFrames** with the same column names as `prices` and a timezone-naive `DatetimeIndex`. `run_backtest` strips timezone automatically (gotchas.md §1), but keeping everything tz-naive from the start is cleaner.

---

## Signal-Driven: MA Crossover

```python
import pandas as pd

ma_fast = features["MA_10"]
ma_slow = features["MA_30"]

# Entry: fast crosses above slow
entries = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))

# Exit: fast crosses below slow
exits = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))

entries_df = pd.DataFrame({"AAPL": entries})
exits_df   = pd.DataFrame({"AAPL": exits})
```

Because features are already shifted by 1 bar at the source (`build_features`), the `.shift(1)` above is a **crossover detection shift** (checking the previous bar's relationship), not a leakage-prevention shift. Both serve different purposes.

---

## ML-Driven: Model Predictions as Signals

```python
# predictions is a 1-D array from model.predict(X_test)
# X_test.index is a DatetimeIndex aligned to the OOS period

entries_df = pd.DataFrame({"AAPL": pd.Series(predictions == 1, index=X_test.index)})
exits_df   = pd.DataFrame({"AAPL": pd.Series(predictions == 0, index=X_test.index)})
```

See `instructions/machine_learning.md` for how to generate `predictions` and `X_test` without lookahead leakage.

---

## Multi-Ticker Signals

Extend the DataFrame to cover every ticker loaded into the `Backtester`:

```python
entries_df = pd.DataFrame({
    "AAPL": aapl_entries,
    "MSFT": msft_entries,
})
exits_df = pd.DataFrame({
    "AAPL": aapl_exits,
    "MSFT": msft_exits,
})
```

Columns that are missing or all-False simply produce no trades for that ticker.
