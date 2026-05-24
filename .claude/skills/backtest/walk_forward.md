# Walk-Forward Validation (rules.md §2)

A single IS/OOS split is insufficient for strategies with optimised parameters — the OOS window may happen to be an easy period. Walk-forward validation simulates how the strategy would have adapted over time by repeatedly re-fitting on expanding or rolling IS windows and testing on the immediately following OOS slice.

---

## Expanding-Window Template

Each iteration adds more IS data and tests the next fixed-length OOS slice:

```python
import pandas as pd
from backtester import p_value_test

windows = [
    # (is_start, is_end, oos_start, oos_end)
    ("2021-01-01", "2022-06-30", "2022-07-01", "2022-09-30"),
    ("2021-01-01", "2022-09-30", "2022-10-01", "2022-12-31"),
    ("2021-01-01", "2022-12-31", "2023-01-01", "2023-03-31"),
    ("2021-01-01", "2023-03-31", "2023-04-01", "2023-06-30"),
]

oos_returns = []

for is_start, is_end, oos_start, oos_end in windows:
    # 1. Fit / optimise strategy parameters on IS window
    #    (e.g. run sensitivity_analysis over is_start → is_end)

    # 2. Generate OOS signals using the best IS parameters
    #    bt = Backtester(tickers=[...], start_date=oos_start, end_date=oos_end, ...)
    #    bt.run_pipeline()
    #    ... build signals ...
    #    pf = bt.run_backtest(entries, exits, freq="D", fees=0.001, slippage=0.001)

    # 3. Collect OOS returns series
    #    oos_returns.append(pf.returns().iloc[:, 0].dropna())
    pass

# 4. Concatenate all OOS slices and run final robustness checks
combined_oos = pd.concat(oos_returns)

result = p_value_test(combined_oos, n_permutations=10_000, random_state=42)
print(f"Walk-forward p-value: {result['p_value']:.4f}")
print(f"Walk-forward Sharpe:  {result['actual_metric']:.3f}")
```

---

## When to Use Walk-Forward

| Situation | Action |
|:---|:---|
| Strategy has ≥ 2 optimised parameters | Required |
| Single fixed rule (e.g. buy-and-hold) | Single IS/OOS split is sufficient |
| ML model with hyperparameter tuning | Required — re-fit model on each IS window |
| OOS Sharpe is significantly below IS Sharpe | Walk-forward will reveal whether the gap is persistent or a single bad period |
