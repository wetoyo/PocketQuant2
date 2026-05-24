# Final Checklist

Use this before marking any backtest as complete. Every box must be checked.

## Data & Leakage
- [ ] Adjusted prices used (`adjusted=True`)
- [ ] Timezone consistent across all data sources
- [ ] Data frequency matches execution window
- [ ] Features consumed from `build_features` — no extra `.shift(1)` applied downstream
- [ ] OOS period was never used for parameter tuning

## Execution
- [ ] Transaction costs included (`fees` and `slippage`)
- [ ] Position size does not exceed 10% of ADV for illiquid names

## Performance (OOS period only)
- [ ] Sharpe Ratio > 1.0
- [ ] Max Drawdown within risk tolerance
- [ ] Profit Factor > 1.3
- [ ] `calculate_alpha_beta` run vs relevant benchmark

## Statistical Robustness
- [ ] `p_value_test` → p-value < 0.05
- [ ] `monte_carlo` → actual Sharpe beats > 80% of shuffled paths
- [ ] `sensitivity_analysis` → Sharpe stays above threshold for ±2 ticks on each parameter
- [ ] Walk-forward validation run (required if ≥ 2 optimised parameters)
