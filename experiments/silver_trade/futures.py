
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta, time

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SilverAnalysis:
    def __init__(self, days=55):
        # 5m data is limited to last 60 days usually for yfinance
        self.days = days
        self.tz = pytz.timezone('US/Eastern')
        
    def fetch_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)
        start_date_extended = start_date - timedelta(days=5) # Extra buffer for SI=F previous days

        logging.info(f"Fetching SI=F and SLV data (Past {self.days} days)...")

        # 1. Fetch SI=F (Silver Futures) - 1h interval for overnight
        # We use 1h to ensure we catch 4pm and 9am nicely, or close to it.
        # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        self.si = yf.download("SI=F", start=start_date_extended, end=end_date, interval="1h", progress=False)
        
        # 2. Fetch SLV (Silver ETF) - 5m interval for opening spike
        self.slv = yf.download("SLV", start=start_date, end=end_date, interval="5m", progress=False)

        # Handle multi-index columns if present (Ticker as level 0)
        self.si = self._clean_df(self.si)
        self.slv = self._clean_df(self.slv)
        
        logging.info(f"Data fetched. SI=F: {len(self.si)} rows, SLV: {len(self.slv)} rows.")

    def _clean_df(self, df):
        if df.empty:
            return df
        # If MultiIndex columns (Price, Ticker), drop Ticker level
        if isinstance(df.columns, pd.MultiIndex):
            # We assume the columns are like (Price, Ticker) and Price is 'Close', 'Open', etc.
            # We want just 'Close', 'Open' etc.
            try:
                # Check if level 1 is ticker
                df.columns = df.columns.get_level_values(0)
            except IndexError:
                pass
        
        # Ensure index is timezone aware and converted to US/Eastern
        if df.index.tz is None:
            # If naive, assume UTC (yfinance usually returns UTC) then convert
            df.index = df.index.tz_localize('UTC')
        
        df.index = df.index.tz_convert(self.tz)
        return df

    def get_price_at_time(self, df, target_time_dt, tolerance_mins=65):
        """
        Finds the price in df closest to target_time_dt within tolerance.
        Returns (price, actual_time) or (None, None).
        """
        # Filter strictly before or at? Or just nearest?
        # Let's look for nearest index within tolerance
        
        # Calculate absolute time difference
        diffs = np.abs(df.index - target_time_dt)
        
        # Find minimum diff
        if len(diffs) == 0:
            return None, None
            
        min_idx = diffs.argmin()
        min_diff = diffs[min_idx]
        
        if min_diff <= timedelta(minutes=tolerance_mins):
            return df.iloc[min_idx]['Close'], df.index[min_idx]
        
        return None, None

    def search_opening_spike(self, date):
        """
        Analyzes SLV opening behavior for a given date.
        Returns dict with stats.
        """
        # Define Open Window: 9:30 AM to 10:00 AM
        open_time = self.tz.localize(datetime.combine(date, time(9, 30)))
        end_spike_window = open_time + timedelta(minutes=30)
        
        # Filter SLV data
        day_data = self.slv[(self.slv.index >= open_time) & (self.slv.index <= end_spike_window)]
        
        if day_data.empty:
            return None
            
        # Opening Price (first bar 9:30)
        # Note: 5m bar at 9:30 covers 9:30-9:35
        try:
            # Get data explicitly at 9:30 if possible, or the first available
            initial_bar = day_data.iloc[0]
            open_price = initial_bar['Open']
            
            # Spike High: Max High in the window
            spike_high = day_data['High'].max()
            spike_time = day_data['High'].idxmax()
            
            spike_pct = (spike_high - open_price) / open_price
            
            return {
                'SLV_Open_Time': day_data.index[0],
                'SLV_Open_Price': open_price,
                'SLV_Spike_High': spike_high,
                'SLV_Spike_Time': spike_time,
                'SLV_Spike_Pct': spike_pct
            }
            
        except Exception as e:
            logging.warning(f"Error calculating spike for {date}: {e}")
            return None

    def calculate_metrics(self):
        results = []
        
        # Iterate over unique dates in SLV
        # Format dates to Timestamp for consistent handling
        slv_dates = sorted(list(pd.to_datetime(self.slv.index.date).unique()))
        
        for i, date_ts in enumerate(slv_dates):
            if i == 0:
                continue # Cannot look back
                
            current_date = date_ts.date() # datetime.date
            prev_date_ts = slv_dates[i-1]
            prev_date = prev_date_ts.date()
            
            # 1. Analyze SLV Opening Spike
            slv_stats = self.search_opening_spike(current_date)
            if not slv_stats:
                continue
                
            # 2. Analyze Overnight SI=F Movement
            # Window: Prev Day 16:00 ET -> Current Day 09:00 ET
             
            # Time definitions
            start_window_target = self.tz.localize(datetime.combine(prev_date, time(16, 0))) # 4 PM ET
            end_window_target = self.tz.localize(datetime.combine(current_date, time(9, 0)))   # 9 AM ET
            
            # Get SI=F prices
            # Use loose tolerance because futures might settle a bit off or data gaps
            si_start_price, si_start_time = self.get_price_at_time(self.si, start_window_target, tolerance_mins=120)
            si_end_price, si_end_time = self.get_price_at_time(self.si, end_window_target, tolerance_mins=60)
            
            if si_start_price is None or si_end_price is None:
                # logging.warning(f"Missing SI=F data for overnight window {prev_date} -> {current_date}")
                continue
                
            si_change_abs = si_end_price - si_start_price
            si_change_pct = si_change_abs / si_start_price
            
            record = {
                'Date': current_date,
                'Prev_Date': prev_date,
                'SI_Start_Time': si_start_time,
                'SI_Start_Price': si_start_price,
                'SI_End_Time': si_end_time,
                'SI_End_Price': si_end_price,
                'SI_Overnight_Pct': si_change_pct,
                **slv_stats
            }
            results.append(record)
            
        self.results_df = pd.DataFrame(results)
        return self.results_df

    def log_results(self):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            logging.warning("No results to log.")
            return

        print("\n--- Analysis Results (Last 5 Rows) ---")
        print(self.results_df[['Date', 'SI_Overnight_Pct', 'SLV_Open_Price', 'SLV_Spike_Pct']].tail())
        
        # Correlation
        corr = self.results_df['SI_Overnight_Pct'].corr(self.results_df['SLV_Spike_Pct'])
        print(f"\nCorrelation between Overnight SI=F % and SLV Opening Spike %: {corr:.4f}")
        
        # Save to CSV
        filename = "expirments/silver_trade/silver_analysis_results.csv"
        self.results_df.to_csv(filename, index=False)
        logging.info(f"Full results saved to {filename}")

if __name__ == "__main__":
    analysis = SilverAnalysis(days=55)
    analysis.fetch_data()
    analysis.calculate_metrics()
    analysis.log_results()
