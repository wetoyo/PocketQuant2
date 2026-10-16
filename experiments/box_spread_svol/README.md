# Box-Spread-Financed SVOL Carry

Backtest of financing an SVOL (NYSEARCA: SVOL) position with a **box-spread loan
on XSP**, timed to month-end so every monthly SVOL distribution is captured,
distributions reinvested, and the loan repaid at expiration by selling SVOL.

Run:

```
.venv\Scripts\python.exe experiments\box_spread_svol\box_spread_svol_backtest.py
```

Outputs (written next to the script): `run_output.txt`, `equity_curves.csv`,
`cycle_ledger.csv`, `sensitivity_grid.csv`, `equity_curve.png`.

---

## The strategy

1. **Borrow via a short box spread on XSP.** A box (bull call spread + bear put
   spread at the same two strikes) has a fixed payoff equal to the strike width,
   so *selling* one is a zero-coupon loan: you receive the discounted width today
   and repay the full width at expiration. XSP (mini-SPX) options are European
   and cash-settled, which makes the box clean and (for retail, via IBKR) one of
   the cheapest USD funding sources available. Historically SPX/XSP box yields
   clear within a few bps of T-bills.
2. **Buy SVOL** with the loan proceeds plus our own equity.
3. **Time everything to the last trading day of the month** so the position is
   always on across SVOL's monthly ex-dividend date.
4. **Reinvest every distribution** into more SVOL on the ex-date.
5. **At each box expiration**, sell just enough SVOL to repay principal + accrued
   box interest, then immediately open a fresh box against the new equity and
   roll on. Rolls run from SVOL inception (2021-05-13) to today.

### Position sizing / what "return" means

The literal ask ("take out a loan, buy SVOL") has no equity base, so a return
series is undefined. The model posts **\$10,000 of our own capital** and borrows
`LEVERAGE x equity` via the box (default `LEVERAGE = 1.0`). Base case therefore
starts with **\$20,000 of SVOL against \$10,000 equity — 2x market exposure** —
and the reported equity curve is `SVOL value + cash - accrued box balance`.
Because the loan notional is fixed for the life of each box, effective leverage
*rises* as SVOL falls (path-dependent, and realistic).

---

## Assumptions & costs

| Item | Value | Notes |
|---|---|---|
| Box financing rate | 13-week T-bill (`^IRX`) + **25 bps** | ranged 0.25% → 5.60% over the test |
| Box interest accrual | simple interest, `L·(1 + r·T)` | T ≈ 0.25 yr; compounding difference negligible |
| Box open cost | **15 bps of notional** | ~4 legs bid/ask + commissions, charged each roll |
| SVOL trade cost | **10 bps per side** (5 fee + 5 slippage) | applied to every buy and sell incl. reinvestment |
| Prices | **raw / unadjusted close** | deliberate — distributions are added back explicitly; using adjusted close would double-count them. SVOL had no splits. |
| Distributions | `Backtester.get_dividends()` (yfinance) | 59 monthly payouts captured |
| Data | yfinance direct for SVOL/SPY/`^IRX` | shared `market_data.db` only caches auto-adjusted OHLC, so the unadjusted pull bypasses `run_pipeline`; robustness still uses `backtester.stats` |

### Not modelled

- Box-spread **assignment / pin risk** (negligible for European XSP) and the
  small mark-to-market swings of an open box before expiry.
- **Margin calls / forced liquidation.** Equity is allowed to run down (and, at
  high leverage, through zero — flagged as `ruined`). A real broker liquidates
  first.
- Section 1256 **tax treatment** of the box (60/40) — this is a pre-tax study.
- Borrow/short-availability constraints on the option legs.
- SVOL's own **expense ratio beyond what is in its NAV** and any tracking noise.

---

## Results — base case (1.0x leverage, 3-month boxes)

| Metric | BoxSVOL | SVOL buy&hold | SPY buy&hold |
|---|--:|--:|--:|
| Total return | **76.3%** | 55.0% | 87.5% |
| CAGR | 11.3% | 8.6% | 12.6% |
| Volatility (ann.) | **54.2%** | 21.6% | 16.9% |
| Max drawdown | **−65.6%** | −33.5% | −25.4% |
| Sharpe | 0.44 | 0.49 | 0.79 |
| Sortino | 0.59 | 0.58 | 1.09 |
| Calmar | 0.17 | 0.26 | 0.50 |

- Final equity **\$17,569** from \$10,000; lowest point along the path **\$6,182**
  (April 2025 volatility spike).
- vs SPY: **beta 2.20, alpha −5.3%/yr**.
- 22 quarterly cycles, **64% win rate**, average cycle **+2.9%**, median **+4.6%**.
  Worst cycle −22.8% (Feb–May 2025).
- **Calendar years:** 2021 +24.7%, 2022 −14.9%, 2023 +40.7%, 2024 +5.8%,
  2025 −2.4%, 2026 +11.1%.
- **IS/OOS (60/40 split, no tuning):** IS Sharpe 0.77 → **OOS Sharpe 0.35**;
  OOS total return +1.1%. The edge is front-loaded in 2021–2023.

### Robustness

- **`p_value_test` / `monte_carlo` (Sharpe): non-informative here.** Both shuffle
  return *order*, but Sharpe and total return are order-invariant, so the null
  distribution collapses to a point (std ≈ 0). Reported for completeness; the
  Sharpe > 0.05 p-value "FAIL" is an artifact of the test, not the strategy.
- **Monte-Carlo drawdown check:** the realised path's −65.6% MaxDD is shallower
  than 57.5% of order-shuffled paths — i.e. an unremarkable ordering, no hidden
  luck in the sequence.
- **Block bootstrap (21-day blocks, 5,000 resamples):** Sharpe 95% CI
  **[−0.16, 1.06]**, `P(Sharpe ≤ 0) = 7.4%`. Wide and straddling zero.
- **Bootstrap vs the honest baseline:** `P(BoxSVOL Sharpe ≤ SVOL-buy&hold Sharpe)
  = 68%` — **box leverage does not improve risk-adjusted return over just holding
  SVOL.**
- **Sensitivity (leverage × box term):** Sharpe is flat at **0.37–0.48** across
  the entire 5×5 grid — no structural sweet spot. Drawdown scales punishingly
  with leverage: −48% at 0.5x, −66% at 1.0x, −83% at 1.5x, **−95% to −100%
  (wipe-out) at 2.0x**. Longer box terms add a little total return (less roll
  cost) with no Sharpe benefit.

---

## Verdict

**The carry is real but the leverage is not worth it.** SVOL's distribution
stream comfortably out-earns the box financing cost (every cycle's dividends >>
its box interest — e.g. \$1,300 vs \$200 in a typical 2024 quarter), so the
levered book does finish ahead of unlevered SVOL on *total return* (76% vs 55%).

But against every risk-adjusted or benchmark test it fails:

- Sharpe **0.44** — below unlevered SVOL (0.49), far below SPY (0.79), and far
  below the skill's 1.0 bar.
- **Negative alpha** vs SPY (−5.3%/yr) at **beta 2.2**.
- A **−66% drawdown** in the April 2025 vol spike (down to \$6,182), exactly the
  short-vol tail that cheap leverage makes existential — and at 2x the position
  is wiped out entirely.
- **OOS Sharpe 0.35** vs IS 0.77: the result leans on the 2021–2023 low-rate,
  calm-vol window.

Box spreads are an efficient *funding* tool; bolting them onto a short-volatility
ETF just buys more of the same tail risk. This is a leverage overlay, not an edge.
