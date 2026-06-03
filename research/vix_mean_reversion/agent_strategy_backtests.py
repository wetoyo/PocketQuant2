import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

from analyzer import compute_signals
from backtest_engine import walk_forward_optimization

def serialize_series(s: pd.Series) -> list:
    """Helper to serialize a pandas Series to a list of dicts with string date keys."""
    return [{'date': k.strftime('%Y-%m-%d'), 'value': float(v)} for k, v in s.items()]

def main():
    print("Agent Strategy Backtests: Starting...")
    
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
    
    # 3. Walk-Forward Optimizations
    print("  Running Walk-Forward for Short VIX...")
    vix_wf = walk_forward_optimization(
        df, spy_df, signal_name_prefix='Sig_Lvl_Z', is_short=True, asset_col='CLOSE'
    )
    
    print("  Running Walk-Forward for Long SPY...")
    spy_long_wf = walk_forward_optimization(
        df, spy_df, signal_name_prefix='Sig_Lvl_Z', is_short=False, asset_col='CLOSE_SPY'
    )
    
    print("  Running Walk-Forward for Short SPY...")
    spy_short_wf = walk_forward_optimization(
        df, spy_df, signal_name_prefix='Sig_Lvl_Z', is_short=True, asset_col='CLOSE_SPY'
    )
    
    # 4. Generate Plot
    plt.figure(figsize=(12, 7))
    plt.plot(vix_wf['test_equity'], label=f"Short VIX (Params: {vix_wf['best_params']})", linewidth=2.0)
    plt.plot(spy_long_wf['test_equity'], label=f"Long SPY (Params: {spy_long_wf['best_params']})", linewidth=2.0)
    plt.plot(spy_short_wf['test_equity'], label=f"Short SPY (Control, Params: {spy_short_wf['best_params']})", linewidth=1.5, linestyle='--')
    plt.title("Walk-Forward Out-of-Sample Equity Curves (2021-2026)", fontsize=15, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Equity (Starting 1.0)", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plots_dir = current_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "strategy_equity_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Export JSON
    output_results = {
        'vix_wf': {
            'best_params': vix_wf['best_params'],
            'train_sharpe': float(vix_wf['train_sharpe']),
            'val_metrics': vix_wf['val_metrics'],
            'test_metrics': vix_wf['test_metrics']
        },
        'spy_long_wf': {
            'best_params': spy_long_wf['best_params'],
            'train_sharpe': float(spy_long_wf['train_sharpe']),
            'val_metrics': spy_long_wf['val_metrics'],
            'test_metrics': spy_long_wf['test_metrics']
        },
        'spy_short_wf': {
            'best_params': spy_short_wf['best_params'],
            'train_sharpe': float(spy_short_wf['train_sharpe']),
            'val_metrics': spy_short_wf['val_metrics'],
            'test_metrics': spy_short_wf['test_metrics']
        }
    }
    
    output_dir = current_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "strategy_backtests.json", "w") as f:
        json.dump(output_results, f, indent=4)
        
    print("Agent Strategy Backtests: Completed successfully.")

if __name__ == "__main__":
    main()
