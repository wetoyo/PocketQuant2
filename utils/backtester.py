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
        """
        if not self.features:
            self._build_features()
            
        if ticker in self.features:
            return self.features[ticker]
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
            **kwargs: Additional arguments for vbt.Portfolio.from_signals (e.g. freq, init_cash, fees).
            
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
        """Get portfolio statistics."""
        if self.portfolio:
            return self.portfolio.stats(**kwargs)
        else:
            print("No portfolio available.")
            return None
