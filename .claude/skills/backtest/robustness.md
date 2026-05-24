# Robustness: Statistical Significance Tests

A strategy that passes performance thresholds but fails these tests is likely overfit or lucky. All three utilities live in `backtester/stats.py` and are exported from `backtester/__init__.py`.

```python
from backtester import p_value_test, monte_carlo, sensitivity_analysis
```

---

## A. P-Value Test (rules.md §5)

Shuffles the return series `n_permutations` times and measures how often the shuffled Sharpe meets or beats the actual Sharpe. A low p-value means the edge is unlikely to be a random artefact of path order.

```python
result = p_value_test(
    portfolio,              # vectorbt Portfolio, pd.Series, or np.ndarray of returns
    n_permutations=10_000,
    metric="sharpe",        # "sharpe" | "total_return"
    periods_per_year=252,   # 252 daily, 52 weekly, 12 monthly
    random_state=42,
)

print(f"P-value:       {result['p_value']:.4f}")
print(f"Actual Sharpe: {result['actual_metric']:.3f}")
print(f"Null median:   {result['median_null']:.3f}  ±  {result['std_null']:.3f}")
```

| Result | Meaning |
|:---|:---|
| `p_value < 0.05` | Significant at 95% confidence ✅ |
| `p_value < 0.01` | Significant at 99% confidence ✅✅ |
| `p_value > 0.10` | Cannot reject null — strategy may be a fluke ❌ |

---

## B. Monte Carlo Simulation (rules.md §5)

Shuffles the returns `n_simulations` times to build a null distribution of Sharpe ratios and max drawdowns. Prints a summary automatically. The actual strategy path is row `simulation_id == -1`.

```python
mc_df = monte_carlo(
    portfolio,
    n_simulations=1_000,
    periods_per_year=252,
    random_state=42,
)

null_df    = mc_df[mc_df["simulation_id"] != -1]
actual_row = mc_df[mc_df["simulation_id"] == -1].iloc[0]

sharpe_pct = (null_df["sharpe"] <= actual_row["sharpe"]).mean()
print(f"Actual Sharpe beats {sharpe_pct:.1%} of random paths")
```

| Result | Meaning |
|:---|:---|
| Actual Sharpe in **top 20%** of paths | Path-independent edge ✅ |
| Actual Sharpe near **median** of paths | No edge beyond lucky ordering ❌ |

---

## C. Sensitivity Analysis (rules.md §5)

Runs the strategy over a grid of parameter combinations. A robust strategy keeps Sharpe above threshold for all neighbours of the chosen parameters — a single-spike surface is overfit.

```python
def run_with_params(params):
    bt2 = Backtester(
        tickers=["AAPL"],
        start_date="2022-01-01",
        end_date="2023-12-31",
        scraper_type="yfinance",
        features_options={"ma_windows": [params["fast"], params["slow"]]}
    )
    bt2.run_pipeline(save_folder_raw=None, save_folder_features=None)
    features2 = bt2.get_ticker_features("AAPL")

    fast = features2[f"MA_{params['fast']}"]
    slow = features2[f"MA_{params['slow']}"]
    e = pd.DataFrame({"AAPL": (fast > slow) & (fast.shift(1) <= slow.shift(1))})
    x = pd.DataFrame({"AAPL": (fast < slow) & (fast.shift(1) >= slow.shift(1))})

    pf = bt2.run_backtest(e, x, freq="D", fees=0.001, slippage=0.001)
    rets = pf.returns().iloc[:, 0].dropna().to_numpy()

    from backtester.stats import _sharpe, _total_return, _max_drawdown
    return {
        "sharpe":       _sharpe(rets),
        "total_return": _total_return(rets),
        "max_drawdown": _max_drawdown(rets),
    }

results = sensitivity_analysis(
    run_with_params,
    param_grid={"fast": [8, 9, 10, 11, 12], "slow": [28, 29, 30, 31, 32]},
)
print(results.sort_values("sharpe", ascending=False).to_string())
```

| Result | Meaning |
|:---|:---|
| Sharpe stays above 1.0 across the grid | Robust ✅ |
| Sharpe collapses at any neighbour of optimal params | Overfit ❌ |
