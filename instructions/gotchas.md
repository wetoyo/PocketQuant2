# Repository Gotchas & Common Pitfalls

This document highlights critical gotchas, architectural assumptions, and common errors to avoid when working with the PocketQuant2 codebase.

---

## 1. Timezone Naive Mismatches in Vectorbt
*   **The Issue**: `vectorbt` requires price DataFrames and signal DataFrames to have identical indexes. The API client cleaners localise raw data datetimes to UTC or convert them to US/Eastern timezone and then strip timezone info (`tz_localize(None)`).
*   **The Gotcha**: If you calculate signals on features that retain timezone info (e.g. `US/Eastern`) and try to pass them to `run_backtest`, vectorbt will fail to align the indexes, throwing:
    ```
    ValueError: No overlapping dates between prices and signals.
    ```
*   **Correction**: Always verify index timezone properties of prices and signals before running a backtest:
    ```python
    prices.index = prices.index.tz_localize(None)
    signals.index = signals.index.tz_localize(None)
    ```

---

## 2. Feature Shift & Lookahead Bias
*   **The Issue**: The target label for machine learning models is whether the *next* price is higher:
    ```python
    df["target"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)
    ```
*   **The Gotcha**: If features $F_t$ (like `MA_10` computed up to time $t$) are directly matched against `target_t`, the model learns to predict price movements using features that have already seen part of the price information at time $t$, causing lookahead bias (leakage).
*   **Correction**: In any predictive pipeline, shift the feature matrix forward by 1 bar:
    ```python
    X = df[feature_cols].shift(1)
    ```

---

## 3. SQLite Database Write Locks
*   **The Issue**: During execution runs or unit tests (`test_scraper_refactor.py`), SQLite files can become locked if multiple database connections are opened and not closed correctly.
*   **The Gotcha**: If a unit test fails or exits abruptly before `conn.close()` is called, the database file will remain locked, and subsequent runs will throw `sqlite3.OperationalError: database is locked` or `PermissionError`.
*   **Correction**: Ensure DB connections are wrapped in `try...finally` blocks or use context managers:
    ```python
    conn = sqlite3.connect(db_path)
    try:
        # Perform operations
    finally:
        conn.close()
    ```

---

## 4. Required Ticker Capitalization
*   **The Issue**: SQLite table names are created from ticker names.
*   **The Gotcha**: The database functions force table names to uppercase:
    ```python
    table_name = ticker.upper()
    ```
    If you pass lowercase tickers to queries, they might not match unless handled, and features CSV files are generated as uppercase names (e.g. `AAPL_features.csv`).
*   **Correction**: Always standardize ticker names to uppercase at entry points.

---

## 5. Options Database Fetch Dates & Weekend Handling
*   **The Issue**: Option chains do not trade on weekends.
*   **The Gotcha**: If option chains are fetched on a Saturday or Sunday, the `FETCH_DATE` written to `options_data.db` could skew comparisons.
*   **Correction**: The `write_options_to_db` helper under `scraper/utils/build_db.py` checks the current weekday:
    *   If Saturday, it sets `FETCH_DATE` to Friday (1 day back).
    *   If Sunday, it sets `FETCH_DATE` to Friday (2 days back).
    Always remember that options data fetched on weekends will be back-dated to Friday.

---

## 6. AWS DynamoDB Billing & Free Tier Limits
*   **The Issue**: Writing high-frequency option chains to AWS DynamoDB can consume substantial storage space.
*   **The Gotcha**: Under [execution/data_fetch/fetch_data_dynamo.py](file:///d:/Files/Code/PocketQuant2/execution/data_fetch/fetch_data_dynamo.py), there is a safety function `table_near_limit` that checks if the table size is near the 25GB free tier limit.
*   **Correction**: The script will abort writes if the table size exceeds 24GB to prevent unexpected AWS bills. If you see writes aborting, you must delete old records using `delete_data_dynamo.py`.

---

## 7. Vectorbt Portfolio Stats Warning
*   **The Issue**: Calling `.stats()` on a multi-column vectorbt Portfolio object without aggregation parameters generates warnings.
*   **The Gotcha**: In newer pandas versions, calling stats on wide DataFrames yields warnings or forces incorrect aggregations.
*   **Correction**: The `Backtester.get_stats()` wrapper explicitly defaults the parameter `agg_func=None`. Maintain this default when fetching stats manually.

---

## 8. Brittle Leakage Test for the `RETURN` Feature (Known Failing Test)

> **Status**: `test_no_future_leakage` in [models/daily_model/xgboost/tests/test_leakage.py](file:///d:/Files/Code/PocketQuant2/models/daily_model/xgboost/tests/test_leakage.py) **fails in CI** (1 of 21 tests).

*   **The Issue**: The test checks that `abs(corr(feature, CLOSE_future)) <= abs(corr(feature, CLOSE_now)) + 1e-3` for every feature. It **always fails for `RETURN`**.
*   **Why It Fails**: `RETURN` is defined as `CLOSE.pct_change()`, which computes the **backward-looking** daily return at time $t$:
    $$\text{RETURN}_t = \frac{C_t - C_{t-1}}{C_{t-1}}$$
    Because stock returns are weakly positively autocorrelated at short horizons, `RETURN_t` is sometimes *slightly* more correlated with $C_{t+1}$ than with $C_t$ itself — a statistical artifact of momentum, not true lookahead leakage. The 1e-3 tolerance in the assertion is too tight.
*   **This is NOT a real data leakage bug**. `RETURN` is computed using only $C_t$ and $C_{t-1}$, so no future information is used. The test's correlation-based heuristic is fundamentally unsuited to detecting leakage in auto-correlated series.
*   **Correction for Agents**: Do not suppress or skip this test, but understand it is a false positive. The correct leakage guard for features is the **shift test** (`test_feature_shift`), which passes. If you add new features, use `test_feature_shift` as the canonical leakage check, not `test_no_future_leakage`.

---

## 9. `test_backtester.py` Returns `bool` Instead of `None` (pytest Warning)

*   **The Issue**: All 10 tests in [tests/test_backtester.py](file:///d:/Files/Code/PocketQuant2/tests/test_backtester.py) return `True` at the end, triggering `PytestReturnNotNoneWarning`.
*   **The Gotcha**: pytest expects test functions to return `None`. Tests that `return True` still pass functionally, but generate noisy warnings in CI output.
*   **Correction**: When writing new tests or extending the existing suite, omit the `return True` statement. The tests in this file were authored as standalone functions (they have their own `run_all_tests` runner) and were later collected by pytest without removing the return values.
