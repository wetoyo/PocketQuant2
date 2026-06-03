import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pytest
import pandas as pd
import numpy as np
from research.vix_mean_reversion.analyzer import (
    compute_signals, calculate_forward_returns, 
    calculate_newey_west_variance, filter_signal_overlap
)

def test_compute_signals():
    # Create dummy VIX data (300 rows)
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
    # Random walk for level
    closes = np.cumsum(np.random.normal(0, 1, 300)) + 20
    closes = np.clip(closes, 10, 80) # Bound VIX
    
    df = pd.DataFrame({'DATE': dates, 'CLOSE': closes})
    df_signals = compute_signals(df)
    
    # Check that output has correct shape and columns
    assert len(df_signals) == 300
    assert 'VIX_RET' in df_signals.columns
    assert 'VIX_LEVEL_Z' in df_signals.columns
    assert 'Sig_Lvl_Z2' in df_signals.columns
    
    # First 251 rows of rolling metrics should have NaNs (since lookback window is 252)
    assert df_signals['VIX_LEVEL_Z'].iloc[0:250].isna().all()
    # Row 252 (index 251) should be populated
    assert not np.isnan(df_signals['VIX_LEVEL_Z'].iloc[251])

def test_calculate_forward_returns():
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    closes = [10.0, 11.0, 12.0, 10.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.0]
    df = pd.DataFrame({'DATE': dates, 'CLOSE': closes})
    
    df_returns = calculate_forward_returns(df, horizons=[1, 2])
    
    # Test 1d forward return at index 0: C_1 / C_0 - 1 = 11 / 10 - 1 = 10%
    assert pytest.approx(df_returns['fwd_ret_1'].iloc[0]) == 0.10
    
    # Test 2d forward return at index 0: C_2 / C_0 - 1 = 12 / 10 - 1 = 20%
    assert pytest.approx(df_returns['fwd_ret_2'].iloc[0]) == 0.20
    
    # Last rows should have NaNs
    assert pd.isna(df_returns['fwd_ret_1'].iloc[-1])
    assert pd.isna(df_returns['fwd_ret_2'].iloc[-2])

def test_newey_west_variance():
    # Simple constant series variance should be 0
    series = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    var_nw = calculate_newey_west_variance(series, lag=1)
    assert pytest.approx(var_nw) == 0.0
    
    # Simple series with variance
    series_2 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    var_nw_2 = calculate_newey_west_variance(series_2, lag=2)
    # NW variance should be positive
    assert var_nw_2 > 0

def test_filter_signal_overlap():
    # Simple signals
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    # Signals on day 1, 2, 3, and 7
    signals = [False, True, True, True, False, False, True, False, False, False]
    df = pd.DataFrame({'DATE': dates, 'Sig': signals})
    
    # Treatment A: Keep all
    sig_a = filter_signal_overlap(df, 'Sig', horizon=2, treatment='A')
    assert (sig_a.values == signals).all()
    
    # Treatment B: Ignore while active (horizon=2)
    # Entry on day 1 (index 1). Open for days 1 and 2. Next signal on day 2 is ignored.
    # Signal on day 3 is entry. Open for days 3 and 4.
    # Signal on day 6 (index 6, day 7) is entry. Open for days 6 and 7.
    sig_b = filter_signal_overlap(df, 'Sig', horizon=2, treatment='B')
    expected_b = [False, True, False, True, False, False, True, False, False, False]
    assert (sig_b.values == expected_b).all()
    
    # Treatment C: First of cluster
    # Clusters: [1, 2, 3] and [6]
    # First of clusters: 1 and 6
    sig_c = filter_signal_overlap(df, 'Sig', horizon=2, treatment='C')
    expected_c = [False, True, False, False, False, False, True, False, False, False]
    assert (sig_c.values == expected_c).all()
    
    # Treatment D: Cooldown of 3 days
    # Entry on day 1 (index 1). Cooldown covers day 1, 2, 3. Next signal is day 6 (index 6).
    sig_d = filter_signal_overlap(df, 'Sig', horizon=2, treatment='D', cooldown_days=3)
    expected_d = [False, True, False, False, False, False, True, False, False, False]
    assert (sig_d.values == expected_d).all()
