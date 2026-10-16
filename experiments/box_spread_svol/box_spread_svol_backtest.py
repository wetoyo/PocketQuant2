"""
Box-Spread-Financed SVOL Carry  —  event-driven backtest
======================================================================

Idea under test
---------------
Borrow cash cheaply by *selling a box spread* on XSP (the mini-SPX,
European, cash-settled, 1256-taxed index option).  A short box is a
zero-coupon loan: you receive the discounted box width today and repay
the full width at expiration.  Historically SPX/XSP box financing has
cleared within a few bps of Treasury bill yields, so we proxy the box
rate as   r_box(t) = 13-week T-bill yield (^IRX) + BOX_SPREAD.

Deploy the borrowed cash (plus our own equity) into SVOL (NYSEARCA:SVOL,
Simplify Volatility Premium ETF), which pays a large *monthly*
distribution.  Everything is timed to the **last trading day of the
month** so that every monthly SVOL distribution is captured while the
position is on.  Distributions are reinvested into SVOL as they arrive.

At each box expiration we sell just enough SVOL to repay the loan
(principal + accrued box interest), then immediately open a fresh box
against the new equity and carry on.  Rolls run from SVOL inception to
today.

What we want to see
-------------------
Does the carry  (SVOL total return  -  box financing cost)  survive
transaction costs and SVOL's short-vol drawdowns once the position is
levered?

Framework notes
---------------
* We use **raw (unadjusted) close** on purpose: this is an explicit
  dividend-reinvestment simulation, so we need raw close + an explicit
  distribution cash stream.  Split/dividend-adjusted prices would
  double-count the distributions we add back by hand.  SVOL had no
  splits over the test window, so raw close is clean.
* SVOL / SPY raw closes and ``^IRX`` are fetched straight from yfinance.
  The shared ``market_data.db`` only caches auto-adjusted OHLC (its
  cached SVOL table has no ``ADJ CLOSE`` column), so routing an
  unadjusted pull through ``Backtester.run_pipeline`` collides with that
  schema — we bypass it for prices and still use the framework's
  ``Backtester.get_dividends`` for the distribution stream and
  ``backtester.stats`` for every robustness test.
* Robustness (p-value, Monte-Carlo, parameter sensitivity) uses
  ``backtester.stats`` on the strategy's daily return series.

Run:
    .venv\\Scripts\\python.exe experiments\\box_spread_svol\\box_spread_svol_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from backtester import Backtester, p_value_test, monte_carlo, sensitivity_analysis
from backtester.stats import _sharpe, _max_drawdown, _total_return

OUT_DIR = Path(__file__).resolve().parent

# ── Strategy parameters ─────────────────────────────────────────────────────
SVOL = "SVOL"
BENCH = "SPY"
START_DATE = "2021-05-12"          # SVOL first trade = 2021-05-13
END_DATE = None                    # -> today

INIT_EQUITY = 10_000.0             # our own capital
LEVERAGE = 1.0                     # box loan notional = LEVERAGE x current equity
BOX_TERM_MONTHS = 3                # months between box roll / repayment
BOX_SPREAD = 0.0025               # financing spread over 13w T-bill (25 bps)
BOX_TXN_COST = 0.0015             # cost to open one box, as fraction of notional (~15 bps)

SVOL_FEE = 0.0005                  # commission per SVOL trade
SVOL_SLIP = 0.0005                # slippage per SVOL trade
SVOL_TC = SVOL_FEE + SVOL_SLIP    # round-number one-way cost applied to every SVOL buy/sell

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════
def _yf_close(ticker: str) -> pd.Series:
    h = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE, auto_adjust=False)
    s = h["Close"].copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.rename(ticker)


def load_data():
    """Return (prices_df[SVOL,SPY] raw close, svol_divs Series, box_rate Series)."""
    prices = pd.concat([_yf_close(SVOL), _yf_close(BENCH)], axis=1).dropna()

    # Framework helper for the distribution stream (tz-naive, values > 0 only).
    bt = Backtester(tickers=[SVOL], start_date=START_DATE, end_date=END_DATE,
                    scraper_type="yfinance", interval="1d", adjusted=False)
    svol_divs = bt.get_dividends()[SVOL].copy()
    svol_divs.index = pd.to_datetime(svol_divs.index)
    if getattr(svol_divs.index, "tz", None) is not None:
        svol_divs.index = svol_divs.index.tz_localize(None)
    svol_divs = svol_divs[svol_divs.index <= prices.index[-1]]

    # 13-week T-bill discount yield, in percent -> decimal.  ffill to every day.
    irx = yf.Ticker("^IRX").history(start=START_DATE, end=END_DATE, auto_adjust=False)["Close"]
    irx.index = pd.to_datetime(irx.index)
    if getattr(irx.index, "tz", None) is not None:
        irx.index = irx.index.tz_localize(None)
    box_rate = (irx / 100.0 + BOX_SPREAD).reindex(prices.index).ffill().bfill()

    return prices, svol_divs, box_rate


# ═══════════════════════════════════════════════════════════════════════════
# Core event-driven simulation
# ═══════════════════════════════════════════════════════════════════════════
def simulate(prices, svol_divs, box_rate,
             leverage=LEVERAGE, box_term_months=BOX_TERM_MONTHS,
             box_spread=BOX_SPREAD, box_txn_cost=BOX_TXN_COST,
             init_equity=INIT_EQUITY):
    """
    Walk day-by-day from SVOL inception:

      * open a box loan of  leverage * equity  on the first day and on
        every roll date (last trading day of the month, every
        `box_term_months` months);
      * reinvest each SVOL distribution into SVOL on its ex-date;
      * on each roll date first repay the maturing box (principal +
        simple box interest for the elapsed term) by selling SVOL, then
        open the next box.

    Returns dict with the daily equity curve and a per-cycle ledger.
    """
    px = prices[SVOL]
    spy = prices[BENCH]
    days = px.index
    first_day, last_day = days[0], days[-1]

    # box_rate carries (13w T-bill + BOX_SPREAD); re-point it if a different
    # spread is requested for a sensitivity run.
    rate = box_rate + (box_spread - BOX_SPREAD)

    # Single roll calendar: last trading day of each month, every Nth month,
    # strictly inside (first_day, last_day).  One trigger, no double rolls.
    month_end_days = pd.DatetimeIndex(sorted(
        pd.Series(days, index=days).groupby([days.year, days.month]).last().values
    ))
    roll_set = {pd.Timestamp(d) for d in month_end_days[::box_term_months]
                if first_day < d < last_day}

    div_map = {pd.Timestamp(d): float(v) for d, v in svol_divs.items()}

    # ── state ──────────────────────────────────────────────────────────────
    cash = init_equity
    shares = 0.0
    loan_principal = 0.0        # cash borrowed and still outstanding
    loan_open = None            # date the live box was opened
    loan_rate = 0.0             # annualised box rate locked at open

    equity_curve = []
    cycle_ledger = []
    cyc = {}                    # running record for the open cycle
    ruined = False

    def accrued_balance(today):
        """Mark-to-model repayment value of the live box as of `today` (simple interest)."""
        if loan_principal <= 0 or loan_open is None:
            return 0.0
        t = (today - loan_open).days / 365.0
        return loan_principal * (1.0 + loan_rate * t)

    def open_box(today, equity_now):
        nonlocal cash, shares, loan_principal, loan_open, loan_rate, cyc
        notional = leverage * max(equity_now, 0.0)
        loan_principal = notional
        loan_open = today
        loan_rate = float(rate.loc[today]) if today in rate.index else float(rate.iloc[-1])

        fee = notional * box_txn_cost
        cash += notional - fee                       # loan proceeds land as cash
        buy_notional = cash                          # deploy every free dollar into SVOL
        shares += buy_notional / (px.loc[today] * (1.0 + SVOL_TC))
        cash -= buy_notional

        cyc = dict(open_date=today, loan_principal=notional, box_rate=loan_rate,
                   box_fee=fee, equity_at_open=equity_now,
                   svol_px_open=float(px.loc[today]), divs_collected=0.0)

    def close_box(today):
        nonlocal cash, shares, loan_principal, loan_open, loan_rate, cyc
        if loan_principal <= 0:
            return None
        repay = accrued_balance(today)
        proceeds_needed = repay - cash
        if proceeds_needed > 0:
            sell_shares = min(proceeds_needed / (px.loc[today] * (1.0 - SVOL_TC)), shares)
            cash += sell_shares * px.loc[today] * (1.0 - SVOL_TC)
            shares -= sell_shares
        cash -= repay                                 # any shortfall shows up as negative equity
        equity_now = shares * px.loc[today] + cash

        cyc.update(close_date=today, svol_px_close=float(px.loc[today]),
                   box_repay=repay, box_interest=repay - cyc["loan_principal"],
                   equity_at_close=equity_now,
                   cycle_return=equity_now / cyc["equity_at_open"] - 1.0)
        cycle_ledger.append(cyc.copy())

        loan_principal = 0.0
        loan_open = None
        loan_rate = 0.0
        return equity_now

    # ── day loop ───────────────────────────────────────────────────────────
    for i, today in enumerate(days):
        # 1. SVOL distribution paid today -> reinvest into SVOL
        if today in div_map and shares > 0:
            div_cash = shares * div_map[today]
            cyc["divs_collected"] = cyc.get("divs_collected", 0.0) + div_cash
            shares += div_cash / (px.loc[today] * (1.0 + SVOL_TC))

        # 2. bootstrap the first box on day 0
        if i == 0:
            open_box(today, init_equity)

        # 3. roll: repay the maturing box, then open a fresh one
        elif today in roll_set:
            eq_now = close_box(today)
            if eq_now is None:
                eq_now = shares * px.loc[today] + cash
            open_box(today, eq_now)

        # 4. mark to market
        equity = shares * px.loc[today] + cash - accrued_balance(today)
        if equity <= 0 and not ruined:
            ruined = True
        equity_curve.append(equity)

    # ── final liquidation ────────────────────────────────────────────────
    close_box(last_day)
    final_equity = shares * px.loc[last_day] * (1.0 - SVOL_TC) + cash
    equity_curve[-1] = final_equity
    if final_equity <= 0:
        ruined = True

    eq = pd.Series(equity_curve, index=days, name="strategy")
    # Floor the curve so a blown-up (ruined) path can't produce nonsense
    # returns when equity crosses zero; `ruined` flags it for the report.
    eq_floor = eq.clip(lower=1e-6)
    ret = eq_floor.pct_change().dropna()

    # benchmarks on the same window --------------------------------------------------
    svol_bh = _buy_hold_reinvest(px, svol_divs, init_equity)
    spy_bh = init_equity * spy / spy.iloc[0]

    return dict(
        equity=eq,
        returns=ret,
        svol_bh=svol_bh,
        spy_bh=spy_bh,
        cycles=pd.DataFrame(cycle_ledger),
        final_equity=final_equity,
        ruined=ruined,
        min_equity=float(eq.min()),
    )


def _buy_hold_reinvest(px, divs, init_equity):
    """Unlevered SVOL with distributions reinvested — the honest baseline."""
    shares = init_equity / px.iloc[0]
    div_map = {pd.Timestamp(d): float(v) for d, v in divs.items()}
    out = []
    for today in px.index:
        if today in div_map:
            shares += shares * div_map[today] / px.loc[today]
        out.append(shares * px.loc[today])
    return pd.Series(out, index=px.index, name="svol_bh")


# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════
def perf_row(equity: pd.Series) -> dict:
    r = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = _sharpe(r.to_numpy(), TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (r.mean() * TRADING_DAYS) / downside if downside > 0 else np.nan
    mdd = _max_drawdown(r.to_numpy())
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return dict(total_return=equity.iloc[-1] / equity.iloc[0] - 1, cagr=cagr,
                vol=vol, sharpe=sharpe, sortino=sortino, max_dd=mdd, calmar=calmar)


def alpha_beta(strat_ret: pd.Series, bench_ret: pd.Series):
    idx = strat_ret.index.intersection(bench_ret.index)
    a, b = strat_ret.loc[idx].to_numpy(), bench_ret.loc[idx].to_numpy()
    beta = np.cov(a, b)[0, 1] / np.var(b)
    alpha_daily = a.mean() - beta * b.mean()
    return beta, alpha_daily * TRADING_DAYS


def fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:>8.2%}"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 74)
    print("  BOX-SPREAD-FINANCED SVOL CARRY  —  backtest since SVOL inception")
    print("=" * 74)

    prices, svol_divs, box_rate = load_data()
    print(f"\nData window : {prices.index[0].date()} -> {prices.index[-1].date()}  "
          f"({len(prices)} trading days)")
    print(f"SVOL distributions captured : {len(svol_divs)}  "
          f"(first {svol_divs.index[0].date()}, last {svol_divs.index[-1].date()})")
    print(f"Box rate range : {box_rate.min():.2%} -> {box_rate.max():.2%}  "
          f"(13w T-bill + {BOX_SPREAD:.2%})")
    print(f"\nParams: leverage={LEVERAGE}x  box_term={BOX_TERM_MONTHS}m  "
          f"box_txn={BOX_TXN_COST:.2%}  svol_tc={SVOL_TC:.2%}/side  init_equity=${INIT_EQUITY:,.0f}")

    res = simulate(prices, svol_divs, box_rate)

    # ── performance table ────────────────────────────────────────────────
    strat = perf_row(res["equity"])
    svolbh = perf_row(res["svol_bh"])
    spybh = perf_row(res["spy_bh"])

    print("\n" + "-" * 74)
    print("PERFORMANCE (since inception, net of costs)")
    print("-" * 74)
    hdr = f"{'':22}{'BoxSVOL':>12}{'SVOL b&h':>12}{'SPY b&h':>12}"
    print(hdr)
    for k, label in [("total_return", "Total return"), ("cagr", "CAGR"),
                     ("vol", "Volatility (ann)"), ("max_dd", "Max drawdown")]:
        print(f"{label:22}{fmt_pct(strat[k]):>12}{fmt_pct(svolbh[k]):>12}{fmt_pct(spybh[k]):>12}")
    for k, label in [("sharpe", "Sharpe"), ("sortino", "Sortino"), ("calmar", "Calmar")]:
        print(f"{label:22}{strat[k]:>12.2f}{svolbh[k]:>12.2f}{spybh[k]:>12.2f}")

    print(f"\nFinal equity: ${res['final_equity']:,.0f}  "
          f"(started ${INIT_EQUITY:,.0f})   min equity along the path: ${res['min_equity']:,.0f}")
    if res["ruined"]:
        print("  !! WARNING: equity hit zero at some point — this path is a wipe-out "
              "(model has no margin-call / forced-liquidation logic).")

    # ── alpha / beta vs SPY ──────────────────────────────────────────────
    spy_ret = res["spy_bh"].pct_change().dropna()
    beta, alpha_ann = alpha_beta(res["returns"], spy_ret)
    print(f"\nvs SPY  ->  beta {beta:.2f}   alpha (ann) {alpha_ann:.2%}")

    # ── per-cycle carry ledger ──────────────────────────────────────────
    cyc = res["cycles"]
    if not cyc.empty:
        show = cyc.copy()
        show["open"] = show["open_date"].dt.date
        show["close"] = show["close_date"].dt.date
        show["loan"] = show["loan_principal"].round(0)
        show["box_rate"] = (show["box_rate"] * 100).round(2)
        show["box_int"] = show["box_interest"].round(0)
        show["divs"] = show["divs_collected"].round(0)
        show["cyc_ret"] = (show["cycle_return"] * 100).round(2)
        print("\n" + "-" * 74)
        print("PER-CYCLE LEDGER (box open -> repay)")
        print("-" * 74)
        print(show[["open", "close", "loan", "box_rate", "box_int", "divs",
                    "equity_at_open", "equity_at_close", "cyc_ret"]]
              .to_string(index=False,
                         formatters={"equity_at_open": lambda v: f"{v:,.0f}",
                                     "equity_at_close": lambda v: f"{v:,.0f}"}))
        win = (cyc["cycle_return"] > 0).mean()
        print(f"\nCycles: {len(cyc)}   win rate: {win:.0%}   "
              f"avg cycle return: {cyc['cycle_return'].mean():.2%}   "
              f"median: {cyc['cycle_return'].median():.2%}")

    # ── calendar-year returns ──────────────────────────────────────────
    print("\n" + "-" * 74)
    print("CALENDAR-YEAR RETURNS")
    print("-" * 74)
    for label, series in [("BoxSVOL", res["equity"]), ("SVOL b&h", res["svol_bh"]),
                          ("SPY b&h", res["spy_bh"])]:
        last = series.resample("YE").last()
        first = series.resample("YE").first()
        yr = (last / first - 1)
        print(f"{label:10} " + "  ".join(f"{i.year}:{v:>7.1%}" for i, v in yr.items()))

    # ── IS / OOS split (60 / 40, no tuning on OOS) ─────────────────────
    split = int(len(res["returns"]) * 0.6)
    is_r = res["returns"].iloc[:split]
    oos_r = res["returns"].iloc[split:]
    print("\n" + "-" * 74)
    print("IN-SAMPLE / OUT-OF-SAMPLE SPLIT  (structural params only; OOS never tuned)")
    print("-" * 74)
    print(f"IS  {is_r.index[0].date()}->{is_r.index[-1].date()}  "
          f"Sharpe {_sharpe(is_r.to_numpy(),TRADING_DAYS):.2f}  "
          f"total {_total_return(is_r.to_numpy()):.2%}")
    print(f"OOS {oos_r.index[0].date()}->{oos_r.index[-1].date()}  "
          f"Sharpe {_sharpe(oos_r.to_numpy(),TRADING_DAYS):.2f}  "
          f"total {_total_return(oos_r.to_numpy()):.2%}")

    # ── robustness ────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("ROBUSTNESS")
    print("=" * 74)

    pv = p_value_test(res["returns"], n_permutations=10_000, metric="sharpe",
                      periods_per_year=TRADING_DAYS, random_state=42)
    print(f"\n[p-value test]  p={pv['p_value']:.4f}  actual Sharpe={pv['actual_metric']:.3f}  "
          f"null median={pv['median_null']:.3f} ± {pv['std_null']:.3f}   "
          f"{'PASS (<0.05)' if pv['p_value'] < 0.05 else 'FAIL'}")

    print("  NOTE: Sharpe and total-return are invariant to return-order permutation,")
    print("        so this shuffle test is degenerate here (null std ~ 0). Treated as")
    print("        non-informative; see block-bootstrap + sensitivity grid instead.")

    mc = monte_carlo(res["returns"], n_simulations=2_000,
                     periods_per_year=TRADING_DAYS, random_state=42)
    null = mc[mc.simulation_id != -1]
    act = mc[mc.simulation_id == -1].iloc[0]
    ddp = (null["max_drawdown"] >= act["max_drawdown"]).mean()
    print(f"[monte carlo]  actual path MaxDD {act['max_drawdown']:.1%} is shallower than "
          f"{ddp:.1%} of order-shuffled paths (path-ordering drawdown check)")

    # ── block bootstrap: does the Sharpe edge survive resampling? ────────
    # Resample the SAME block indices from both return series so the
    # BoxSVOL-minus-SVOL Sharpe gap is measured on matched days.
    common = res["returns"].index.intersection(
        res["svol_bh"].pct_change().dropna().index)
    rr = res["returns"].reindex(common).to_numpy()
    sv = res["svol_bh"].pct_change().reindex(common).to_numpy()
    rng = np.random.default_rng(42)
    bl = 21                                  # ~1-month blocks keep short-horizon autocorr
    n_boot = 5_000
    n = len(rr)
    idx_pool = np.arange(n - bl)
    boot_sharpe = np.empty(n_boot)
    boot_gap = np.empty(n_boot)              # BoxSVOL Sharpe minus SVOL-b&h Sharpe
    for b in range(n_boot):
        starts = rng.choice(idx_pool, size=n // bl + 1)
        sel = np.concatenate([np.arange(s, s + bl) for s in starts])[:n]
        boot_sharpe[b] = _sharpe(rr[sel], TRADING_DAYS)
        boot_gap[b] = boot_sharpe[b] - _sharpe(sv[sel], TRADING_DAYS)
    lo, hi = np.percentile(boot_sharpe, [2.5, 97.5])
    p_pos = float((boot_sharpe <= 0).mean())
    p_beats_svol = float((boot_gap <= 0).mean())
    print(f"\n[block bootstrap, {bl}d blocks x {n_boot}]  "
          f"Sharpe 95% CI [{lo:.2f}, {hi:.2f}]   P(Sharpe<=0)={p_pos:.3f}")
    print(f"[block bootstrap]  P(BoxSVOL Sharpe <= SVOL-buy&hold Sharpe) = {p_beats_svol:.3f}   "
          f"{'levering ADDS risk-adjusted value' if p_beats_svol < 0.05 else 'levering does NOT beat plain SVOL on risk-adjusted basis'}")

    # ── sensitivity: leverage x box term ─────────────────────────────
    print("\n[sensitivity]  leverage x box_term_months")

    def run_params(p):
        r = simulate(prices, svol_divs, box_rate,
                     leverage=p["leverage"], box_term_months=p["term"])
        rr = r["returns"].to_numpy()
        return dict(sharpe=_sharpe(rr, TRADING_DAYS),
                    total_return=_total_return(rr),
                    max_drawdown=_max_drawdown(rr),
                    final_equity=r["final_equity"],
                    ruined=r["ruined"])

    grid = sensitivity_analysis(run_params,
                                {"leverage": [0.5, 0.75, 1.0, 1.5, 2.0],
                                 "term": [1, 2, 3, 6, 12]})
    print("\nSharpe by leverage (rows) x box_term_months (cols):")
    print(grid.pivot(index="leverage", columns="term", values="sharpe").round(2).to_string())
    print("\nMax drawdown:")
    print(grid.pivot(index="leverage", columns="term", values="max_drawdown").round(3).to_string())
    print("\nTotal return:")
    print(grid.pivot(index="leverage", columns="term", values="total_return").round(2).to_string())
    if grid["ruined"].any():
        blown = grid[grid["ruined"]][["leverage", "term"]].values.tolist()
        print(f"\nWipe-out (equity hit zero) at leverage/term: {blown}")

    # ── persist artefacts ────────────────────────────────────────────
    curves = pd.DataFrame({
        "strategy": res["equity"],
        "svol_buy_hold": res["svol_bh"],
        "spy_buy_hold": res["spy_bh"],
    })
    curves.to_csv(OUT_DIR / "equity_curves.csv")
    if not cyc.empty:
        cyc.to_csv(OUT_DIR / "cycle_ledger.csv", index=False)
    grid.to_csv(OUT_DIR / "sensitivity_grid.csv", index=False)
    print(f"\nSaved: equity_curves.csv, cycle_ledger.csv, sensitivity_grid.csv  -> {OUT_DIR}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 9), height_ratios=[3, 1])
        curves.plot(ax=ax[0], logy=True, lw=1.3)
        ax[0].set_title("Box-spread-financed SVOL carry vs baselines (log scale)")
        ax[0].set_ylabel("Equity ($)")
        ax[0].grid(True, alpha=.3)
        dd = res["equity"] / res["equity"].cummax() - 1
        ax[1].fill_between(dd.index, dd.values, 0, color="firebrick", alpha=.4)
        ax[1].set_ylabel("Drawdown")
        ax[1].grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "equity_curve.png", dpi=110)
        print(f"Saved: equity_curve.png -> {OUT_DIR}")
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
