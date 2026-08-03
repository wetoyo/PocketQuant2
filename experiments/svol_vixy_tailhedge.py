"""
SVOL + VIXY Tail Hedge Backtest (event-driven)
======================================================================
Strategy:
  - Start: May 12, 2021 (SVOL inception). Initial cash = $10,000 → all into SVOL.
  - Each SVOL dividend date:
      1. Collect SVOL dividend income (shares × div/share that day).
      2. Check SVOL 1-month momentum and MACD histogram:
         - DOWN (negative momentum or histogram < 0) → route dividend cash into VIXY.
         - UP/flat → reinvest dividend cash into SVOL.
      3. VIXY dividend income is always reinvested into VIXY.
  - No leverage, long-only.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType, Direction

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtester import Backtester

# ── Constants ────────────────────────────────────────────────────────────────
START_DATE   = "2021-05-12"
END_DATE     = None
INITIAL_CASH = 10_000.0
SVOL = "SVOL"
VIXY = "VIXY"

features_options = {"macd_params": [(12,26,9)]}

# ── Initialize Backtester ───────────────────────────────────────────────────
bt = Backtester(
    tickers=[SVOL, VIXY],
    start_date=START_DATE,
    end_date=END_DATE,
    features_options=features_options
)
bt.run_pipeline()

# ── Fetch dividends ─────────────────────────────────────────────────────────
svol_divs = bt.get_dividends()[SVOL]  # pd.Series, ex-dividend dates
print(svol_divs)
vixy_divs = bt.get_dividends()[VIXY]

# ── Fetch prices ────────────────────────────────────────────────────────────
prices = bt.get_price_data()[[SVOL, VIXY]]

# ── Compute hedging signal ─────────────────────────────────────────────────
features = bt.get_ticker_features(SVOL)
macd = features['MACD_12_26_9']
signal = features['MACD_SIGNAL_12_26_9']
hist = macd - signal

# calculate monthly signal (1 month prior)
monthly_mom = prices[SVOL].pct_change(21).resample("ME").last()
monthly_hist = (macd - signal).resample("ME").last()
route_to_vixy_monthly = (monthly_mom < 0) | (monthly_hist < 0)

# Align signal to dividend dates (event-driven)
div_dates = svol_divs.index
route_to_vixy_div = route_to_vixy_monthly.reindex(div_dates, method="ffill")

# ── Simulation ─────────────────────────────────────────────────────────────
order_size = pd.DataFrame(0.0, index=prices.index, columns=[SVOL, VIXY])

# initial buy (all SVOL)
first_price = prices[SVOL].iloc[0]
svol_shares = INITIAL_CASH / first_price
vixy_shares = 0.0
cash = 0.0
order_size.loc[prices.index[0], SVOL] = svol_shares

# event-driven dividend loop
for dt in div_dates:
    if dt not in prices.index:
        # if ex-div date is missing in price (holiday), use next available date
        dt = prices.index[prices.index.get_loc(dt, method='bfill')]
    
    # dividend cash
    div_cash = svol_shares * svol_divs.loc[dt]
    if div_cash <= 0:
        continue
    
    # decide where to reinvest
    if route_to_vixy_div.loc[dt]:
        # buy VIXY
        buy_shares = div_cash / prices.loc[dt, VIXY]
        vixy_shares += buy_shares
        order_size.loc[dt, VIXY] += buy_shares
    else:
        # reinvest in SVOL
        buy_shares = div_cash / prices.loc[dt, SVOL]
        svol_shares += buy_shares
        order_size.loc[dt, SVOL] += buy_shares
    
    # reinvest any VIXY dividends automatically
    vixy_div_cash = vixy_shares * vixy_divs.reindex([dt]).fillna(0).iloc[0]
    if vixy_div_cash > 0:
        buy_shares_vixy = vixy_div_cash / prices.loc[dt, VIXY]
        vixy_shares += buy_shares_vixy
        order_size.loc[dt, VIXY] += buy_shares

# ── Build portfolio ─────────────────────────────────────────────────────────
pf = vbt.Portfolio.from_orders(
    close=prices,
    size=order_size,
    size_type=SizeType.Amount,
    direction=Direction.LongOnly,
    init_cash=INITIAL_CASH,
    freq="D"  # daily frequency
)

# ── Output ────────────────────────────────────────────────────────────────
print(pf.stats())
pf.value().vbt.plot()
print("SVOL buys per month:")
print(order_size[SVOL][order_size[SVOL] > 0])

print("\nVIXY buys per month:")
print(order_size[VIXY][order_size[VIXY] > 0])