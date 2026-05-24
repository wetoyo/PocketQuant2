# Execution: Running the Backtest with Realistic Costs

A backtest without transaction costs is not a valid result (rules.md §3). Always include `fees` and `slippage`.

---

## Signal-Driven Execution

```python
portfolio = bt.run_backtest(
    entries_df,
    exits_df,
    freq="D",           # Match your data interval: "D", "H", "5min", etc.
    init_cash=10_000,
    fees=0.001,         # 0.1% per trade (typical retail broker)
    slippage=0.001,     # 0.1% slippage model
)
```

`run_backtest` wraps `vbt.Portfolio.from_signals`. Any additional vectorbt kwargs (e.g. `direction`, `size`, `group_by`) are passed through.

---

## Event-Driven Execution (Order-Based)

Use this for strategies where you calculate exact share quantities at specific events (e.g. dividend reinvestment, rebalancing). See `instructions/backtesting.md §4` for full details.

```python
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction

portfolio = vbt.Portfolio.from_orders(
    close=prices,
    size=order_sizes,           # DataFrame of share quantities (zeros = no action)
    size_type=SizeType.Amount,
    direction=Direction.LongOnly,
    init_cash=10_000,
    freq="D",
)
```

---

## Transaction Cost Reference (rules.md §3)

| Cost type | Typical value | Parameter |
|:---|:---|:---|
| Broker commission | 0.05%–0.10% per trade | `fees` |
| Slippage — liquid stocks | 0.05%–0.15% per trade | `slippage` |
| Slippage — illiquid / small-cap | 0.5%–2.0% per trade | `slippage` |
| Short borrowing cost | 0.5%–5% annually | Model separately |

**Liquidity rule**: Do not execute a trade larger than 10% of the average daily volume. vectorbt does not enforce this — filter signals manually if position size is material.
