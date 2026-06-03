import sqlite3
import json
import pandas as pd
from pathlib import Path
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

from analyzer import compute_signals, calculate_forward_returns, get_stats_table, filter_signal_overlap

def main():
    print("Agent Overlap Analysis: Starting...")
    
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
    
    # 2. Compute Signals and Returns
    df = compute_signals(df)
    df = calculate_forward_returns(df, [10])
    
    rep_sig = 'Sig_Lvl_Z2'
    treatments = ['A', 'B', 'C', 'D']
    
    results = {}
    for treat in treatments:
        filtered_sig = filter_signal_overlap(df, rep_sig, horizon=10, treatment=treat, cooldown_days=10)
        ret_10 = df.loc[filtered_sig == True, 'fwd_ret_10'].dropna()
        stats_10 = get_stats_table(ret_10, horizon_lag=10)
        results[treat] = stats_10
        
    # Save Results JSON
    output_dir = current_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "overlap_analysis.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Agent Overlap Analysis: Completed successfully.")

if __name__ == "__main__":
    main()
