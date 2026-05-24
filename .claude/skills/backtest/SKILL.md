---
name: backtesting
description: Use when creating, evaluating, validating, or stress-testing quantitative trading backtests in PocketQuant2.
---

# Skill: How to Create a Backtest

Follow the steps below in order. Each links to a focused reference file. Do not skip Step 6 or 7 — a strategy that only passes Steps 1–5 is not complete.

---

## Steps

| # | What | File |
|:---|:---|:---|
| 0 | Verify data quality and leakage-prevention rules before writing any code | [preflight.md](preflight.md) |
| 1–3 | Initialize `Backtester`, run the data pipeline, retrieve prices and features | [setup.md](setup.md) |
| 4 | Generate boolean entry and exit signals (MA crossover or ML predictions) | [signals.md](signals.md) |
| 5 | Execute `run_backtest` with realistic fees and slippage | [execution.md](execution.md) |
| 6 | Read portfolio stats and compute Alpha/Beta vs a benchmark | [evaluation.md](evaluation.md) |
| 7 | Run p-value test, Monte Carlo, and sensitivity analysis | [robustness.md](robustness.md) |
| 8 | Walk-forward validation for strategies with optimised parameters | [walk_forward.md](walk_forward.md) |
| ✓ | Final checklist before marking the backtest complete | [checklist.md](checklist.md) |

---

## Key Rules (from `instructions/rules.md`)

- **Features are pre-shifted at the source.** `build_features` applies `.shift(1)` automatically. Never re-apply it in a pipeline or model.
- **Always include transaction costs.** A backtest with `fees=0` and `slippage=0` is not a valid result.
- **OOS data is sacred.** Never tune parameters against the OOS window, even once.
- **Three robustness gates must pass:** p-value < 0.05, Monte Carlo Sharpe in top 20%, sensitivity surface is smooth.

---

## Relevant Source Files

| File | Purpose |
|:---|:---|
| `backtester/backtester.py` | `Backtester` class — all pipeline, signal, and portfolio methods |
| `backtester/stats.py` | `p_value_test`, `monte_carlo`, `sensitivity_analysis` |
| `instructions/rules.md` | Full rule rationale and thresholds |
| `instructions/backtesting.md` | `run_backtest` / `from_orders` API reference |
| `instructions/evaluation_metrics.md` | Alpha/Beta math, feature evaluation, delay decay |
| `instructions/machine_learning.md` | XGBoost pipeline, leakage prevention, chrono split |
| `instructions/examples.md` | Copy-paste templates for MA, ML, and event-driven backtests |
| `instructions/gotchas.md` | Known pitfalls and their fixes |
