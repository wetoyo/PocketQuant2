import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators and boolean shock signals for VIX.
    Expects df with columns: DATE, CLOSE (VIX level).
    """
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Calculate daily VIX returns
    df['VIX_RET'] = df['CLOSE'].pct_change()
    
    # Rolling 252-day lookback metrics to avoid lookahead bias
    roll = df['VIX_RET'].rolling(252)
    df['VIX_RET_P95'] = roll.quantile(0.95)
    df['VIX_RET_P99'] = roll.quantile(0.99)
    
    df['VIX_RET_MEAN'] = roll.mean()
    df['VIX_RET_STD'] = roll.std()
    df['VIX_RET_Z'] = (df['VIX_RET'] - df['VIX_RET_MEAN']) / df['VIX_RET_STD']
    
    roll_lvl = df['CLOSE'].rolling(252)
    df['VIX_LEVEL_P95'] = roll_lvl.quantile(0.95)
    df['VIX_LEVEL_MEAN'] = roll_lvl.mean()
    df['VIX_LEVEL_STD'] = roll_lvl.std()
    df['VIX_LEVEL_Z'] = (df['CLOSE'] - df['VIX_LEVEL_MEAN']) / df['VIX_LEVEL_STD']
    
    # Define boolean event flags
    df['Sig_Ret_P95'] = (df['VIX_RET'] > df['VIX_RET_P95']).astype(bool)
    df['Sig_Ret_P99'] = (df['VIX_RET'] > df['VIX_RET_P99']).astype(bool)
    df['Sig_Lvl_P95'] = (df['CLOSE'] > df['VIX_LEVEL_P95']).astype(bool)
    df['Sig_Lvl_Z2'] = (df['VIX_LEVEL_Z'] > 2.0).astype(bool)
    df['Sig_Lvl_Z3'] = (df['VIX_LEVEL_Z'] > 3.0).astype(bool)
    df['Sig_Ret_Z2'] = (df['VIX_RET_Z'] > 2.0).astype(bool)
    df['Sig_Ret_Z3'] = (df['VIX_RET_Z'] > 3.0).astype(bool)
    
    return df

def calculate_forward_returns(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    """
    Computes forward VIX returns for given horizons: R_{t, H} = C_{t+H} / C_t - 1.
    Note that this does NOT shift signals, it calculates forward return at index t.
    """
    df = df.copy()
    for h in horizons:
        df[f'fwd_ret_{h}'] = df['CLOSE'].shift(-h) / df['CLOSE'] - 1.0
    return df

def calculate_newey_west_variance(series: pd.Series, lag: int) -> float:
    """
    Estimates Newey-West variance of the series to account for autocorrelation in overlapping returns.
    """
    n = len(series)
    if n <= lag + 1:
        return float(series.var()) if n > 1 else 0.0
    
    mean = series.mean()
    # Lag 0 autocovariance (standard variance)
    gamma = [np.mean((series.values - mean) ** 2)]
    
    # Autocovariances up to lag
    for j in range(1, lag + 1):
        cov = np.mean((series.values[j:] - mean) * (series.values[:-j] - mean))
        gamma.append(cov)
        
    var_nw = gamma[0]
    for j in range(1, lag + 1):
        weight = 1.0 - (j / (lag + 1.0))
        var_nw += 2.0 * weight * gamma[j]
        
    # Scale for degree of freedom adjustment (N / (N - k))
    var_nw = var_nw * (n / (n - 1))
    return float(var_nw)

def get_stats_table(returns: pd.Series, horizon_lag: int) -> Dict:
    """
    Computes statistical metrics for a series of forward returns, including
    standard and Newey-West adjusted t-stats and 95% confidence intervals.
    """
    n = len(returns)
    if n < 2:
        return {
            'Count': n, 'Mean': np.nan, 'Median': np.nan, 'StdDev': np.nan,
            'WinRate_Short': np.nan, 't_stat': np.nan, 'p_val': np.nan,
            't_stat_NW': np.nan, 'p_val_NW': np.nan,
            'CI_Lower': np.nan, 'CI_Upper': np.nan,
            'CI_Lower_NW': np.nan, 'CI_Upper_NW': np.nan
        }
        
    mean = float(returns.mean())
    median = float(returns.median())
    std = float(returns.std())
    
    # Win rate for short position (VIX drops -> return is negative)
    win_rate_short = float((returns < 0).sum() / n)
    
    # Standard t-statistic and p-value
    t_stat, p_val = stats.ttest_1samp(returns.values, 0.0)
    se_std = std / np.sqrt(n)
    ci_lower = mean - 1.96 * se_std
    ci_upper = mean + 1.96 * se_std
    
    # Newey-West t-statistic and p-value
    var_nw = calculate_newey_west_variance(returns, horizon_lag)
    std_nw = np.sqrt(max(var_nw, 1e-10))
    se_nw = std_nw / np.sqrt(n)
    
    t_stat_nw = mean / se_nw if se_nw > 0 else np.nan
    
    # Two-sided p-value from t-distribution
    if not np.isnan(t_stat_nw):
        p_val_nw = 2.0 * (1.0 - stats.t.cdf(abs(t_stat_nw), df=n - 1))
    else:
        p_val_nw = np.nan
        
    ci_lower_nw = mean - 1.96 * se_nw
    ci_upper_nw = mean + 1.96 * se_nw
    
    return {
        'Count': n,
        'Mean': mean,
        'Median': median,
        'StdDev': std,
        'WinRate_Short': win_rate_short,
        't_stat': float(t_stat),
        'p_val': float(p_val),
        't_stat_NW': float(t_stat_nw),
        'p_val_NW': float(p_val_nw),
        'CI_Lower': float(ci_lower),
        'CI_Upper': float(ci_upper),
        'CI_Lower_NW': float(ci_lower_nw),
        'CI_Upper_NW': float(ci_upper_nw)
    }

def filter_signal_overlap(df: pd.DataFrame, signal_col: str, horizon: int, treatment: str, cooldown_days: int = 10) -> pd.Series:
    """
    Applies overlap filters to raw signals.
    Returns a boolean series containing the filtered signal flags.
    
    Treatments:
      - 'A': Keep all signals (parallel positions)
      - 'B': Ignore new signals while a position is open (holding period = horizon)
      - 'C': First signal in a cluster of consecutive signal days
      - 'D': Cooldown period of N days after a signal
    """
    signals = df[signal_col].values
    n = len(signals)
    filtered = np.zeros(n, dtype=bool)
    
    if treatment == 'A':
        return pd.Series(signals, index=df.index)
        
    elif treatment == 'B':
        active_until = -1
        for i in range(n):
            if signals[i]:
                if i >= active_until:
                    filtered[i] = True
                    active_until = i + horizon
                    
    elif treatment == 'C':
        for i in range(n):
            if signals[i]:
                if i == 0 or not signals[i-1]:
                    filtered[i] = True
                    
    elif treatment == 'D':
        active_until = -1
        for i in range(n):
            if signals[i]:
                if i >= active_until:
                    filtered[i] = True
                    active_until = i + cooldown_days
                    
    return pd.Series(filtered, index=df.index)

def split_regimes(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
    """
    Constructs regime indices for historical periods, market trends (SPY), and VIX levels.
    """
    df = df.copy()
    dates = pd.to_datetime(df['DATE'])
    
    # 1. Historical Periods
    regimes = {
        'Pre-2008': dates < '2008-01-01',
        '2008-Crisis': (dates >= '2008-01-01') & (dates <= '2009-06-30'),
        '2009-2019': (dates >= '2009-07-01') & (dates <= '2019-12-31'),
        'COVID-Crash': (dates >= '2020-01-01') & (dates <= '2020-12-31'),
        'Post-COVID': dates >= '2021-01-01'
    }
    
    # 2. Bull vs Bear markets (requires SPY)
    if spy_df is not None:
        spy_aligned = pd.merge(df[['DATE']], spy_df[['DATE', 'CLOSE']], on='DATE', how='left')
        spy_aligned['CLOSE'] = spy_aligned['CLOSE'].ffill().bfill()
        spy_aligned['MA200'] = spy_aligned['CLOSE'].rolling(200).mean()
        
        regimes['Bull-Market'] = (spy_aligned['CLOSE'] > spy_aligned['MA200']).values
        regimes['Bear-Market'] = (spy_aligned['CLOSE'] <= spy_aligned['MA200']).values
    else:
        regimes['Bull-Market'] = np.ones(len(df), dtype=bool)
        regimes['Bear-Market'] = np.zeros(len(df), dtype=bool)
        
    # 3. High-volatility vs Low-volatility VIX regimes
    # High vol: VIX > rolling 252-day median
    vix_roll_median = df['CLOSE'].rolling(252).median()
    regimes['High-Vol'] = (df['CLOSE'] > vix_roll_median).values
    regimes['Low-Vol'] = (df['CLOSE'] <= vix_roll_median).values
    
    # Convert all to pandas boolean Series
    return {k: pd.Series(v, index=df.index) for k, v in regimes.items()}

def bootstrap_confidence_interval(returns: pd.Series, n_boot: int = 10000) -> Tuple[float, float, float]:
    """
    Computes the bootstrap mean, 95% confidence interval, and p-value (H0: mean >= 0).
    """
    n = len(returns)
    if n < 5:
        return np.nan, np.nan, np.nan
        
    boot_means = np.zeros(n_boot)
    vals = returns.values
    for i in range(n_boot):
        boot_means[i] = np.mean(np.random.choice(vals, size=n, replace=True))
        
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    
    # Bootstrap p-value (probability of mean >= 0 under empirical distribution)
    # Note: we test if mean is negative, so p-value is fraction of bootstrap means >= 0.
    p_val_boot = float((boot_means >= 0.0).sum() / n_boot)
    
    return float(np.mean(boot_means)), ci_lower, ci_upper
