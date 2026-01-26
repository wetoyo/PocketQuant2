import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add project root to path for imports
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from scraper.setup import BaseSetup
from configs.paths import DATA_FEATURES

class FeatureEvaluator:
    def __init__(self, df: pd.DataFrame, ticker: str):
        """
        Initialize with a DataFrame containing 'CLOSE', 'DATE', and feature columns.
        """
        self.df = df.copy()
        self.ticker = ticker
        self.features = []
        self.results = {}

        if 'DATE' in self.df.columns:
            self.df['DATE'] = pd.to_datetime(self.df['DATE'])
            self.df = self.df.sort_values('DATE').reset_index(drop=True)
        
        # Identify features (exclude non-feature columns)
        exclude = ['DATE', 'TICKER', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'INTERVAL', 'RETURN', 'LOG_RETURN']
        self.features = [c for c in self.df.columns if c not in exclude and not c.endswith('_q')]

    def compute_forward_returns(self, horizons: List[int] = [5, 15]):
        """
        Computes forward returns for given horizons.
        fwd_ret_n = (Close_{t+n} / Close_t) - 1
        """
        for h in horizons:
            col_name = f'fwd_ret_{h}'
            self.df[col_name] = self.df['CLOSE'].shift(-h) / self.df['CLOSE'] - 1
        return self.df

    def bin_features(self, n_quantiles: int = 10):
        """
        Creates quantile bins for each feature.
        """
        for feat in self.features:
            try:
                # Use rank to handle duplicates more gracefully
                self.df[f'{feat}_q'] = pd.qcut(self.df[feat].rank(method='first'), n_quantiles, labels=False)
            except Exception as e:
                print(f"Error binning {feat}: {e}")

    def evaluate_feature(self, feature: str, horizon_col: str):
        """
        Computes conditional expectation of forward returns given feature quantiles.
        """
        q_col = f'{feature}_q'
        if q_col not in self.df.columns or horizon_col not in self.df.columns:
            return None

        # Drop NaNs for the specific evaluation
        temp_df = self.df[[q_col, horizon_col]].dropna()
        if temp_df.empty:
            return None

        grouped = temp_df.groupby(q_col)[horizon_col].agg(['mean', 'std', 'count']).reset_index()
        grouped.columns = ['quantile', 'mean_ret', 'std_ret', 'count']
        return grouped

    def plot_monotonicity(self, feature: str, horizon_col: str, results_df: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Plots mean forward return by feature quantile.
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x='quantile', y='mean_ret', data=results_df, palette='RdYlGn', hue='quantile', legend=False)
        plt.title(f'{self.ticker} | Forward Return ({horizon_col}) by {feature} Quantile', fontsize=14)
        plt.xlabel(f'{feature} Quantile', fontsize=12)
        plt.ylabel('Mean Forward Return', fontsize=12)
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        plt.grid(alpha=0.2)
        
        if save_path:
            plt.savefig(save_path / f"{self.ticker}_{feature}_{horizon_col}_monotonicity.png")
            plt.close()
        else:
            plt.show()

    def analyze_regimes(self, feature: str, horizon_col: str, save_path: Optional[Path] = None):
        """
        Analyze feature performance across different regimes: Time of Day and Volatility.
        """
        if 'DATE' not in self.df.columns:
            return

        # 1. Time of Day
        self.df['hour'] = self.df['DATE'].dt.hour
        self.df['time_regime'] = 'Midday'
        self.df.loc[self.df['hour'] < 11, 'time_regime'] = 'Open'
        self.df.loc[self.df['hour'] >= 15, 'time_regime'] = 'Close'

        # 2. Volatility Regime
        if 'RETURN' not in self.df.columns:
            self.df['RETURN'] = self.df['CLOSE'].pct_change()
        
        self.df['vol_rolling'] = self.df['RETURN'].rolling(20).std()
        try:
            self.df['vol_regime'] = pd.qcut(self.df['vol_rolling'], 3, labels=['Low', 'Medium', 'High'])
        except Exception:
            self.df['vol_regime'] = 'Medium'

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Time of Day
        sns.barplot(ax=axes[0], x=f'{feature}_q', y=horizon_col, hue='time_regime', data=self.df, palette='muted')
        axes[0].set_title( f'Time of Day Regime: {feature}')
        axes[0].axhline(0, color='black', linestyle='--')

        # Volatility
        sns.barplot(ax=axes[1], x=f'{feature}_q', y=horizon_col, hue='vol_regime', data=self.df, palette='coolwarm')
        axes[1].set_title(f'Volatility Regime: {feature}')
        axes[1].axhline(0, color='black', linestyle='--')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path / f"{self.ticker}_{feature}_{horizon_col}_regimes.png")
            plt.close()
        else:
            plt.show()

    def evaluate_delay(self, feature: str, horizon_col: str, delays=[0, 1, 2], save_path: Optional[Path] = None):
        """
        Evaluate signal decay with execution delay.
        """
        delay_results = []
        for d in delays:
            # Shift the horizon return backward relative to the feature (simulating entering late)
            delayed_ret = self.df[horizon_col].shift(-d)
            temp_df = pd.DataFrame({
                'q': self.df[f'{feature}_q'],
                'fwd_ret': delayed_ret
            }).dropna()
            
            if temp_df.empty: continue
            
            stats = temp_df.groupby('q')['fwd_ret'].mean()
            # Spread between highest and lowest quantile
            spread = stats.max() - stats.min() 
            delay_results.append({'delay': d, 'spread': spread})
        
        delay_df = pd.DataFrame(delay_results)
        
        plt.figure(figsize=(8, 5))
        sns.lineplot(x='delay', y='spread', data=delay_df, marker='o', linewidth=2)
        plt.title(f'Delay Decay: {feature} (Spread top-bottom)')
        plt.ylabel('Return Spread')
        plt.xlabel('Bars of Delay')
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path / f"{self.ticker}_{feature}_{horizon_col}_delay.png")
            plt.close()
        else:
            plt.show()
        return delay_df

    def classify_feature(self, feature: str, results_5: pd.DataFrame, delay_df: pd.DataFrame) -> Dict:
        """
        Categorize feature based on evaluation results.
        """
        if results_5 is None or results_5.empty:
            return {}

        # 1. Directionality
        # Check mean returns of first and last quantiles
        q_min_ret = results_5.iloc[0]['mean_ret']
        q_max_ret = results_5.iloc[-1]['mean_ret']
        
        # Simplistic: if last quantile > first, it's generally momentum-like for that indicator
        # (Assuming positive indicator values usually mean "up" or "overbought")
        direction = "Momentum" if q_max_ret > q_min_ret else "Mean Reversion"
        
        # 2. Strength (Spread)
        spread = abs(q_max_ret - q_min_ret)
        strength = "Strong" if spread > 0.001 else "Medium" if spread > 0.0005 else "Weak"
        
        # 3. Tradability (Delay Sensitivity)
        # If spread drops more than 50% in 1 bar, it's fragile
        base_spread = delay_df[delay_df['delay'] == 0]['spread'].iloc[0] if not delay_df.empty else 0
        d1_spread = delay_df[delay_df['delay'] == 1]['spread'].iloc[0] if len(delay_df) > 1 else 0
        
        tradability = "Realistic"
        if base_spread > 0 and (d1_spread / base_spread) < 0.5:
            tradability = "Fragile"

        return {
            "feature": feature,
            "direction": direction,
            "strength": strength,
            "spread": spread,
            "tradability": tradability
        }

def run_evaluation_system(tickers: List[str], start_date: str = "2025-01-01", interval: str = "1m", feature_options: Optional[Dict] = None):
    """
    Main entry point for evaluating features across multiple tickers.
    """
    if feature_options is None:
        feature_options = {
            "returns": True,
            "log_returns": True,
            "ma_windows": [5, 20, 50],
            "bb_windows": [20],
            "rsi_windows": [14],
            "macd_params": [(12, 26, 9)],
            "vol_windows": [10],
            "atr_windows": [14]
        }

    # 1. Initialize Scraper Setup
    setup = BaseSetup(
        tickers=tickers,
        start_date=start_date,
        interval=interval,
        features_options=feature_options
    )
    
    # Run full pipeline to get data and features
    print(f"--- Fetching Data and Generating Features for {tickers} ---")
    setup.run_pipeline()
    
    # Base results directory
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    results_base = Path("research/results")
    results_base.mkdir(parents=True, exist_ok=True)
    
    inventory = []

    for ticker in tickers:
        print(f"\n>>> Evaluating Features for {ticker}...")
        
        # Create ticker-specific directory
        ticker_dir = results_base / f"{today_str}_{ticker}"
        plot_dir = ticker_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Get raw data and features for this ticker
        # note: setup.data is {ticker: raw_df}, setup.features is {ticker: feat_df}
        raw_df = setup.data.get(ticker)
        feat_df = setup.features.get(ticker)
        
        if raw_df is None or feat_df is None:
            print(f"Skipping {ticker}: No data/features available.")
            continue

        # Merge for evaluation
        # We need DATE and CLOSE from raw, and everything from feat
        combined_df = pd.merge(
            raw_df[['DATE', 'CLOSE']], 
            feat_df, 
            on='DATE', 
            how='inner'
        )
        
        evaluator = FeatureEvaluator(combined_df, ticker)
        evaluator.compute_forward_returns(horizons=[5, 15])
        evaluator.bin_features(n_quantiles=5)

        # Evaluate and Classify each feature
        ticker_inventory = []
        for feat in evaluator.features:
            print(f"Testing {feat}...")
            
            # Forward return horizon of 5 bars
            results_5 = evaluator.evaluate_feature(feat, 'fwd_ret_5')
            if results_5 is None: continue
            
            # Delay sensitivity
            delay_df = evaluator.evaluate_delay(feat, 'fwd_ret_5', delays=[0, 1, 2], save_path=plot_dir)
            
            # Classification
            classification = evaluator.classify_feature(feat, results_5, delay_df)
            classification['ticker'] = ticker
            ticker_inventory.append(classification)
            inventory.append(classification)
            
            # Plots (only for "Strong" or "Medium" to avoid noise explosion, or all for MVP)
            if classification['strength'] in ['Strong', 'Medium']:
                evaluator.plot_monotonicity(feat, 'fwd_ret_5', results_5, save_path=plot_dir)
                evaluator.analyze_regimes(feat, 'fwd_ret_5', save_path=plot_dir)
        
        # Save ticker-specific inventory
        if ticker_inventory:
            ticker_inv_df = pd.DataFrame(ticker_inventory)
            csv_path = ticker_dir / f"{ticker}_features.csv"
            ticker_inv_df.sort_values(by='spread', ascending=False).to_csv(csv_path, index=False)
            print(f"Saved results for {ticker} to {csv_path}")

    # Global Inventory Report (optional, kept for summary)
    inventory_df = pd.DataFrame(inventory)
    print("\n" + "="*50)
    print("FEATURE EVALUATION SUMMARY")
    print("="*50)
    if not inventory_df.empty:
        print(inventory_df.sort_values(by='spread', ascending=False).to_string(index=False))
        # Also save comprehensive one in the base results dir if desired, 
        # or just rely on the individual ones. 
        # For now, let's keep the legacy global file in research root or move it to results_base
        # global_csv_path = results_base / f"{today_str}_full_inventory.csv"
        # inventory_df.to_csv(global_csv_path, index=False)
    else:
        print("No features evaluated successfully.")

if __name__ == "__main__":
    # Example run
    run_evaluation_system(["AAPL"], start_date="2025-12-27", interval="1m")
