import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

from analyzer import compute_signals, calculate_forward_returns, get_stats_table

def main():
    print("Agent Event Study: Starting...")
    
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
    
    # 3. Phase 1: Event Study
    horizons = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60]
    df = calculate_forward_returns(df, horizons)
    
    signal_cols = {
        'Sig_Ret_P95': 'VIX Return > 95th Pctl',
        'Sig_Ret_P99': 'VIX Return > 99th Pctl',
        'Sig_Lvl_P95': 'VIX Level > 95th Pctl',
        'Sig_Lvl_Z2': 'VIX Z-Score > 2.0',
        'Sig_Lvl_Z3': 'VIX Z-Score > 3.0',
        'Sig_Ret_Z2': 'VIX Return Z-Score > 2.0',
        'Sig_Ret_Z3': 'VIX Return Z-Score > 3.0'
    }
    
    results = {}
    for sig, label in signal_cols.items():
        sig_mask = df[sig] == True
        results[sig] = {
            'label': label,
            'count': int(sig_mask.sum()),
            'horizons': {}
        }
        
        for h in horizons:
            fwd_ret = df.loc[sig_mask, f'fwd_ret_{h}'].dropna()
            stats_dict = get_stats_table(fwd_ret, horizon_lag=h)
            results[sig]['horizons'][str(h)] = stats_dict
            
    # 4. Phase 4: Decay Curve
    all_horizons = list(range(1, 61))
    decay_df = calculate_forward_returns(df, all_horizons)
    
    # Generate plot
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="darkgrid")
    
    decay_paths = {}
    for sig, label in signal_cols.items():
        sig_mask = decay_df[sig] == True
        path_means = []
        for h in all_horizons:
            mean_val = float(decay_df.loc[sig_mask, f'fwd_ret_{h}'].mean())
            path_means.append(mean_val)
            
        plt.plot(all_horizons, [x * 100 for x in path_means], label=f"{label} (N={sig_mask.sum()})", linewidth=2.5)
        decay_paths[sig] = path_means
        
    plt.axhline(0, color='black', linestyle='--', linewidth=1.2)
    plt.title("Average VIX Return Path After Volatility Shocks", fontsize=15, fontweight='bold')
    plt.xlabel("Days Post-Shock", fontsize=12)
    plt.ylabel("Average Cumulative VIX Return (%)", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plots_dir = current_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "vix_decay_path.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Results JSON
    output_dir = current_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "event_study.json", "w") as f:
        json.dump({
            'event_study': results,
            'decay_paths': decay_paths
        }, f, indent=4)
        
    print("Agent Event Study: Completed successfully.")

if __name__ == "__main__":
    main()
