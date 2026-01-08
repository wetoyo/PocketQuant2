import pandas as pd
import vectorbt as vbt
from scraper.setup import BaseSetup

class Backtester(BaseSetup):
    """
    Backtesting class that inherits from BaseSetup.
    Uses vectorbt for backtesting strategies on the scraped data.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.portfolio = None

    def get_price_data(self, price_col="CLOSE"):
        """
        Reshapes the self.data dict into a single DataFrame with tickers as columns.
        Useful for vectorbt.
        """
        if not self.data:
            print("No data found in memory. Checking if pipeline needs to be run...")
            # If data is empty, we might need to fetch it. 
            # But BaseSetup doesn't auto-fetch. 
            # We can check if we have data in DB or just run pipeline.
            # Let's try to fetch from DB/API if empty.
            self._fetch_data()
            
        combined_df = pd.DataFrame()
        
        for ticker, df in self.data.items():
            if df.empty:
                continue
            
            # Ensure DATE is index
            temp_df = df.copy()
            if "DATE" in temp_df.columns:
                temp_df["DATE"] = pd.to_datetime(temp_df["DATE"])
                temp_df.set_index("DATE", inplace=True)
            elif not isinstance(temp_df.index, pd.DatetimeIndex):
                # Try to convert index
                try:
                    temp_df.index = pd.to_datetime(temp_df.index)
                except:
                    print(f"Could not convert index to datetime for {ticker}")
                    continue
            
            # Select price column
            if price_col in temp_df.columns:
                combined_df[ticker] = temp_df[price_col]
            else:
                print(f"Column {price_col} not found for {ticker}. Available: {temp_df.columns.tolist()}")
        
        return combined_df

    def get_ticker_data(self, ticker):
        """
        Get data for a specific ticker.
        """
        if ticker in self.data:
            return self.data[ticker]
        return None

    def get_ticker_features(self, ticker):
        """
        Get features for a specific ticker.
        Returns a DataFrame with datetime index if DATE column exists.
        """
        if not self.features:
            self._build_features()
            
        if ticker in self.features:
            features = self.features[ticker].copy()
            
            # Convert DATE column to datetime index if it exists
            if "DATE" in features.columns:
                features = features.set_index(pd.to_datetime(features["DATE"]))
            elif not isinstance(features.index, pd.DatetimeIndex):
                # Try to convert existing index to datetime
                try:
                    features.index = pd.to_datetime(features.index)
                except:
                    print(f"Could not convert index to datetime for {ticker} features")
            
            return features
        return None

    def align_signals(self, signals_dict):
        """
        Align signals from a dictionary {ticker: series/array} into a DataFrame.
        """
        return pd.DataFrame(signals_dict)

    def run_backtest(self, entries, exits, price_col="CLOSE", **kwargs):
        """
        Run a backtest using vectorbt.
        
        Args:
            entries (pd.DataFrame): Boolean DataFrame of entry signals (tickers as columns).
            exits (pd.DataFrame): Boolean DataFrame of exit signals (tickers as columns).
            price_col (str): Column name to use for execution price (default "CLOSE").
            **kwargs: Additional arguments for vbt.Portfolio.from_signals, such as:
                - freq (str or pd.Timedelta): Frequency of the data. 
                    - "D" for daily
                    - "H" for hourly
                    - "min" for minute (5min, 15min, 30min)
                    - "W" for weekly
                    - "M" for monthly
                - init_cash (float): Initial cash for the portfolio.
                - fees (float): Trading fees per transaction.
                - slippage (float): Slippage per trade.
                - size (float or pd.Series): Position sizing.
                - direction (str): 'long', 'short', or 'both'.
                - group_by (str or list): Columns to group by for multiple portfolios.

            
        Returns:
            vbt.Portfolio: The resulting portfolio object.
        """
        prices = self.get_price_data(price_col=price_col)
        
        # Align prices with signals
        # Ensure indices match
        common_index = prices.index.intersection(entries.index)
        if common_index.empty:
            raise ValueError("No overlapping dates between prices and signals.")
            
        prices = prices.loc[common_index]
        entries = entries.loc[common_index]
        exits = exits.loc[common_index]
        
        # Run vectorbt backtest
        self.portfolio = vbt.Portfolio.from_signals(
            prices, 
            entries, 
            exits, 
            **kwargs
        )
        return self.portfolio

    def plot_portfolio(self, **kwargs):
        """Plot the portfolio results."""
        if self.portfolio:
            return self.portfolio.plot(**kwargs)
        else:
            print("No portfolio to plot. Run backtest first.")

    def get_stats(self, **kwargs):
        """Get portfolio statistics.
        
        Args:
            **kwargs: Additional arguments for portfolio.stats(), such as:
                - agg_func: Aggregation function for multi-column portfolios (default: None for separate stats)
                - metrics: List of specific metrics to return
        """
        if self.portfolio:
            # Set agg_func=None by default to avoid aggregation warning
            # This returns stats for each column separately
            if 'agg_func' not in kwargs:
                kwargs['agg_func'] = None
            return self.portfolio.stats(**kwargs)
        else:
            print("No portfolio available.")
            return None

    def calculate_alpha_beta(self, benchmark_ticker, strategy_ticker, freq="D", error_tolerance=0.20, date_range=None):
        """
        Calculate Alpha and Beta against a benchmark using asset returns, with lookback window
        determined by the target relative standard error.
        
        Args:
            benchmark_ticker (str): Ticker symbol for the benchmark (e.g., "QQQ").
            strategy_ticker (str): Ticker symbol for the strategy asset (e.g., "NVDA").
            freq (str): Frequency to resample to for calculation (default "D" for daily). Can be '5min', '15min', etc.
            error_tolerance (float): Target relative standard error for Beta (default 0.20).
            date_range (tuple): Optional (start_date, end_date) to restrict calculation period.
            
        Returns:
            dict: {'beta', 'alpha', 'alpha_per_period', 'n_bars_used', 'required_n', 'beta_full'}
        """
        if self.portfolio is None:
            print("No portfolio available. Run backtest first.")
            return None

        try:
            # --- 1. Get prices ---
            prices_df = self.get_price_data(price_col="CLOSE")
            if strategy_ticker not in prices_df.columns:
                print(f"Strategy ticker {strategy_ticker} not found in price data.")
                return None
            if benchmark_ticker not in prices_df.columns:
                print(f"Benchmark {benchmark_ticker} not found in price data.")
                return None

            strategy_prices = prices_df[strategy_ticker]
            benchmark_prices = prices_df[benchmark_ticker]

            # --- 2. Apply date range filter ---
            if date_range:
                start_date, end_date = date_range
                strategy_prices = strategy_prices[start_date:end_date]
                benchmark_prices = benchmark_prices[start_date:end_date]

            # --- 3. Resample if frequency is specified ---
            if freq:
                strategy_prices = strategy_prices.resample(freq).last().dropna()
                benchmark_prices = benchmark_prices.resample(freq).last().dropna()

            # --- 4. Compute returns ---
            strategy_rets = strategy_prices.pct_change().dropna()
            benchmark_rets = benchmark_prices.pct_change().dropna()

            # --- 5. Align indices ---
            idx = strategy_rets.index.intersection(benchmark_rets.index)
            if len(idx) < 30:
                print(f"Not enough overlapping data ({len(idx)} bars) for reliable alpha/beta calculation.")
                return None

            strat_full = strategy_rets.loc[idx]
            bench_full = benchmark_rets.loc[idx]

            # --- 6. Full beta and residuals ---
            cov_full = strat_full.cov(bench_full)
            var_full = bench_full.var()
            if var_full == 0:
                print("Benchmark variance is zero.")
                return None

            beta_full = cov_full / var_full
            alpha_full = strat_full.mean() - beta_full * bench_full.mean()
            residuals = strat_full - (alpha_full + beta_full * bench_full)
            sigma_epsilon = residuals.std(ddof=1)
            sigma_m = bench_full.std(ddof=1)

            # --- 7. Compute required N for error tolerance ---
            abs_beta = max(abs(beta_full), 1e-6)  # avoid div by zero
            target_se = error_tolerance * abs_beta
            required_n = int((sigma_epsilon / (sigma_m * target_se)) ** 2)
            required_n = min(required_n, len(strat_full))
            required_n = max(required_n, 30)  # minimum bars

            # --- 8. Use last N bars ---
            strat_subset = strat_full.iloc[-required_n:]
            bench_subset = bench_full.iloc[-required_n:]

            cov_subset = strat_subset.cov(bench_subset)
            var_subset = bench_subset.var()
            beta_final = cov_subset / var_subset if var_subset != 0 else 0

            # --- 9. Alpha per period ---
            alpha_per_period = strat_subset.mean() - beta_final * bench_subset.mean()

            # --- 10. Annualize alpha based on frequency ---
            freq_to_periods = {
                "D": 252,
                "W": 52,
                "M": 12
            }
            periods_per_year = freq_to_periods.get(freq, 252)  # default to daily
            alpha_annualized = alpha_per_period * periods_per_year

            return {
                "beta": beta_final,
                "alpha": alpha_annualized,
                "alpha_per_period": alpha_per_period,
                "n_bars_used": required_n,
                "required_n": required_n,
                "beta_full": beta_full
            }

        except Exception as e:
            print(f"Error calculating alpha/beta: {e}")
            import traceback
            traceback.print_exc()
            return None