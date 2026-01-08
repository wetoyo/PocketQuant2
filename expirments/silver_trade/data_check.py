import yfinance as yf
import pandas as pd
import datetime

# 1. Check SI=F (Silver Futures)
print("--- SI=F Data Check ---")
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=7)

# Fetch 1h data to see overnight coverage
si = yf.download("SI=F", start=start_date, end=end_date, interval="60m", progress=False)
print("SI=F shape:", si.shape)
if not si.empty:
    print(si.head())
    print(si.tail())
    print("Unique hours present in index:", si.index.hour.unique())

# 2. Check SLV (Silver ETF)
print("\n--- SLV Data Check ---")
# Fetch 5m data to see market open headers
slv = yf.download("SLV", start=start_date, end=end_date, interval="5m", progress=False)
print("SLV shape:", slv.shape)
if not slv.empty:
    print(slv.head())
    print(slv.tail())
    # Check for 9:30 AM ET data (which would be 14:30 UTC usually, or depends on localization)
    print("Unique hours present in index:", slv.index.hour.unique())
