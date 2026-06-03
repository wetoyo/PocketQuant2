import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

from analyzer import compute_signals, calculate_forward_returns, get_stats_table, split_regimes, bootstrap_confidence_interval

def main():
    print("Agent Regime Analysis: Starting...")
    
    # 1. Load Data
    db_path = project_root / "data" / "database" / "market_data.db"
    conn = sqlite3.connect(str(db_path))
    vix_df = pd.read_sql('SELECT * FROM "^VIX" WHERE INTERVAL = "1d"', conn)
    spy_df = pd.read_sql('SELECT * FROM "SPY" WHERE INTERVAL = "1d"', conn)
    conn.close()
    
    vix_df['DATE'] = pd.to_datetime(vix_df['DATE'])
    spy_df['DATE'] = pd.to_datetime(spy_df['DATE'])
    
    df = pd.merge(vix_df[['DATE', 'CLOSE']], spy_df[['DATE', 'CLOSE']], on='DATE', suffixes=('_VIX', '_SPY'), how='inner')
    df = df.rename(columns={'CLOSE_VIX': 'CLOSE', 'CLOSE_SPY': 'CLOSE_SPY'})
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # 2. Compute Signals
    df = compute_signals(df)
    
    # Compute forward returns for key horizons
    horizons = [5, 10, 20]
    df = calculate_forward_returns(df, horizons)
    
    rep_sig = 'Sig_Lvl_Z2'
    
    # 3. Phase 2: Regime Analysis
    regimes = split_regimes(df, spy_df=spy_df)
    regime_results = {}
    
    for r_name, r_mask in regimes.items():
        combined_mask = (df[rep_sig] == True) & r_mask
        cnt = int(combined_mask.sum())
        
        regime_results[r_name] = {
            'count': cnt,
            'horizons': {}
        }
        
        if cnt < 5:
            continue
            
        for h in horizons:
            fwd_ret = df.loc[combined_mask, f'fwd_ret_{h}'].dropna()
            stats_dict = get_stats_table(fwd_ret, horizon_lag=h)
            regime_results[r_name]['horizons'][str(h)] = stats_dict
            
    # 4. Phase 5: Robustness Testing
    rep_returns = df.loc[df[rep_sig] == True, 'fwd_ret_10'].dropna()
    boot_mean, boot_ci_lower, boot_ci_upper = bootstrap_confidence_interval(rep_returns)
    
    # Subsample split
    split_date = '2008-01-01'
    sub1 = df[(df['DATE'] < split_date) & (df[rep_sig] == True)]['fwd_ret_10'].dropna()
    sub2 = df[(df['DATE'] >= split_date) & (df[rep_sig] == True)]['fwd_ret_10'].dropna()
    
    stats_sub1 = get_stats_table(sub1, horizon_lag=10)
    stats_sub2 = get_stats_table(sub2, horizon_lag=10)
    
    robustness_results = {
        'bootstrap': {
            'observed_mean': float(rep_returns.mean()),
            'bootstrap_mean': boot_mean,
            'ci_lower': boot_ci_lower,
            'ci_upper': boot_ci_upper,
            'failure_rate': float((rep_returns >= 0).sum() / len(rep_returns))
        },
        'subsample_1': stats_sub1,
        'subsample_2': stats_sub2
    }
    
    # Save Results JSON
    output_dir = current_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "regime_analysis.json", "w") as f:
        json.dump({
            'regime_analysis': regime_results,
            'robustness': robustness_results
        }, f, indent=4)
        
    print("Agent Regime Analysis: Completed successfully.")

if __name__ == "__main__":
    main()
