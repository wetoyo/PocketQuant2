# PocketQuant2 — Operational Manual for AI Agents

This is the top-level index for the repository documentation located in `/instructions/`. Each file is a self-contained, implementation-focused reference. Start here to locate the information you need.

---

## Quick Reference: File Index

| File | What It Covers |
|:---|:---|
| [architecture.md](file:///d:/Files/Code/PocketQuant2/instructions/architecture.md) | Directory layout, class hierarchy, data flow diagram, module map |
| [data_pipeline.md](file:///d:/Files/Code/PocketQuant2/instructions/data_pipeline.md) | API clients, SQLite schema, caching logic, feature computation formulas |
| [backtesting.md](file:///d:/Files/Code/PocketQuant2/instructions/backtesting.md) | Signal-driven and event-driven backtest workflows, signal alignment |
| [evaluation_metrics.md](file:///d:/Files/Code/PocketQuant2/instructions/evaluation_metrics.md) | Portfolio stats, Alpha/Beta math, feature evaluation, delay decay |
| [machine_learning.md](file:///d:/Files/Code/PocketQuant2/instructions/machine_learning.md) | XGBoost pipeline, target labeling, leakage prevention, chrono split |
| [examples.md](file:///d:/Files/Code/PocketQuant2/instructions/examples.md) | Minimal MA crossover, ML classifier, event-driven dividend backtest |
| [skills_backtest.md](file:///d:/Files/Code/PocketQuant2/instructions/skills_backtest.md) | End-to-end backtest guide: setup → signals → costs → stats → robustness tests |
| [gotchas.md](file:///d:/Files/Code/PocketQuant2/instructions/gotchas.md) | Known bugs, test failures, timezone pitfalls, DB locking, CI issues |

---

## Key Entry Points by Task

### "I want to run a new backtest"
1. Read [skills_backtest.md](file:///d:/Files/Code/PocketQuant2/instructions/skills_backtest.md) — the end-to-end guide (setup → signals → costs → stats → robustness)
2. Copy a template from [examples.md](file:///d:/Files/Code/PocketQuant2/instructions/examples.md)
3. For detailed API reference: [backtesting.md](file:///d:/Files/Code/PocketQuant2/instructions/backtesting.md)
4. If confused by date mismatches: [gotchas.md](file:///d:/Files/Code/PocketQuant2/instructions/gotchas.md) → **Gotcha #1**

### "I want to add a new technical indicator/feature"
1. Read [data_pipeline.md](file:///d:/Files/Code/PocketQuant2/instructions/data_pipeline.md) → **Section 3** (feature table)
2. Add the compute function in `scraper/features/your_feature.py`
3. Export it from [`scraper/features/__init__.py`](file:///d:/Files/Code/PocketQuant2/scraper/features/__init__.py)
4. Register it in [`scraper/utils/feature_builder.py`](file:///d:/Files/Code/PocketQuant2/scraper/utils/feature_builder.py) → `build_features()`
5. Add the parameter key to `features_options` in `BaseSetup.__init__`

### "I want to train an ML model"
1. Read [machine_learning.md](file:///d:/Files/Code/PocketQuant2/instructions/machine_learning.md) fully
2. Pay close attention to **Section 2** (leakage prevention via `.shift(1)`)
3. Use [`models/daily_model/xgboost/xgboost_model.py`](file:///d:/Files/Code/PocketQuant2/models/daily_model/xgboost/xgboost_model.py) as a template
4. Run the test suite in `models/daily_model/xgboost/tests/` to verify correctness

### "I want to evaluate a feature's predictive signal"
1. Read [evaluation_metrics.md](file:///d:/Files/Code/PocketQuant2/instructions/evaluation_metrics.md) → **Section 3**
2. Use `FeatureEvaluator` in [`research/evaluate_features.py`](file:///d:/Files/Code/PocketQuant2/research/evaluate_features.py)
3. Call `run_evaluation_system(["AAPL"], start_date="...", interval="1m")`

### "I want to compute Alpha/Beta vs a benchmark"
1. Read [evaluation_metrics.md](file:///d:/Files/Code/PocketQuant2/instructions/evaluation_metrics.md) → **Section 2**
2. Both the strategy ticker AND benchmark ticker must be loaded into the same `Backtester` instance
3. A portfolio (`run_backtest`) must be executed first — `calculate_alpha_beta` returns `None` otherwise

### "Tests are failing / CI is broken"
1. Read [gotchas.md](file:///d:/Files/Code/PocketQuant2/instructions/gotchas.md) fully
2. **`test_no_future_leakage` failure on `RETURN`** is a known false positive — see **Gotcha #8**
3. **`PytestReturnNotNoneWarning`** warnings from `test_backtester.py` are cosmetic — see **Gotcha #9**

---

## Repository Test Suite Summary

Run with: `.\.venv\Scripts\python.exe -m pytest`

| Test File | Tests | Status |
|:---|:---:|:---:|
| `expirments/test_av.py` | 2 | ✅ Pass |
| `models/daily_model/xgboost/tests/test_auc.py` | 3 | ✅ Pass |
| `models/daily_model/xgboost/tests/test_labels.py` | 1 | ✅ Pass |
| `models/daily_model/xgboost/tests/test_leakage.py` | 2 | ⚠️ 1 known false positive |
| `models/daily_model/xgboost/tests/test_split.py` | 1 | ✅ Pass |
| `tests/test_backtester.py` | 10 | ✅ Pass (with warnings) |
| `tests/test_real_aapl.py` | 1 | ✅ Pass |
| `tests/test_scraper_refactor.py` | 1 | ✅ Pass |
| **Total** | **21** | **20 pass / 1 known false positive** |

---

## Canonical Data Shapes

An AI agent working with this codebase should always expect the following shapes:

| Object | Type | Shape / Schema |
|:---|:---|:---|
| `bt.data` | `dict` | `{ "TICKER": pd.DataFrame(DATE, OPEN, HIGH, LOW, CLOSE, VOLUME, INTERVAL) }` |
| `bt.features` | `dict` | `{ "TICKER": pd.DataFrame(DATE, MA_n, RSI_n, MACD_..., ..., TICKER) }` |
| `bt.get_price_data()` | `pd.DataFrame` | `DatetimeIndex × [TICKER1, TICKER2, ...]`, all CLOSE prices |
| `entries` / `exits` | `pd.DataFrame` | `DatetimeIndex × [TICKER1, TICKER2, ...]`, boolean dtype |
| `bt.portfolio` | `vbt.Portfolio` | vectorbt Portfolio object (access via `.stats()`, `.total_return()`, etc.) |
| `bt.get_dividends()` | `dict` | `{ "TICKER": pd.Series(DatetimeIndex → float) }`, only nonzero dividend dates |

---

## Important File Locations

| Path | Purpose |
|:---|:---|
| [`configs/paths.py`](file:///d:/Files/Code/PocketQuant2/configs/paths.py) | All project path constants (`DATA_RAW`, `DATABASE_PATH`, etc.) |
| `data/database/market_data.db` | SQLite cache for price data (one table per ticker) |
| `data/database/options_data.db` | SQLite cache for options chains |
| `data/features/{TICKER}_features.csv` | Serialized feature CSVs (generated by pipeline) |
| `execution/tickers.txt` | Ticker list used by scheduled daily fetch scripts |
| `execution/logs/` | Log files from execution runs (timestamped) |
