# Evaluation: Performance Metrics and Alpha/Beta

---

## Core Portfolio Metrics

```python
stats = bt.get_stats()
print(stats)
```

`get_stats()` calls `portfolio.stats(agg_func=None)` — the `agg_func=None` default prevents pandas aggregation warnings on multi-ticker portfolios (gotchas.md §7).

### Minimum Acceptance Thresholds (rules.md §4)

| Metric | Minimum bar | Strong |
|:---|:---|:---|
| Sharpe Ratio | > 1.0 | > 1.5 |
| Sortino Ratio | > 1.0 | > 2.0 |
| Max Drawdown | > −30% | > −15% |
| Calmar Ratio | > 0.5 | > 1.0 |
| Profit Factor | > 1.3 | > 1.7 |
| Win Rate | context-dependent — pair with Win/Loss ratio | |

A high win rate is meaningless if the average loss significantly exceeds the average win (rules.md §4).

---

## Alpha and Beta vs Benchmark

Both the strategy ticker **and** the benchmark ticker must be loaded into the same `Backtester` instance. `run_backtest` must have been called first — `calculate_alpha_beta` returns `None` otherwise.

```python
# Benchmark must be in bt.tickers when Backtester was initialized
metrics = bt.calculate_alpha_beta(
    benchmark_ticker="SPY",
    strategy_ticker="AAPL",
    freq="D",
    error_tolerance=0.20,   # 20% relative standard error tolerance on Beta
)

print(f"Beta:          {metrics['beta']:.2f}")
print(f"Alpha (ann.):  {metrics['alpha']:.2%}")
print(f"N bars used:   {metrics['n_bars_used']} / required: {metrics['required_n']}")
```

The method uses an adaptive lookback window — it computes how many bars are needed to achieve the target standard error on Beta before reporting the final estimate. See `instructions/evaluation_metrics.md §2` for the full math.

### Optional: Restrict to a Date Range

```python
metrics = bt.calculate_alpha_beta(
    benchmark_ticker="SPY",
    strategy_ticker="AAPL",
    freq="D",
    date_range=("2023-01-01", "2023-06-30"),  # OOS period only
)
```
