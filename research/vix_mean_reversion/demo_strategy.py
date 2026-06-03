import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

from analyzer import compute_signals, filter_signal_overlap

def main():
    print("====================================================")
    
    print("DEMO: VIX-Spike-Driven SPY Long Trading Strategy")
    print("Optimal Parameters:")
    print("  - Signal: VIX Level Z-Score > 3.0")
    print("  - Holding Period (H): 3 Trading Days")
    print("  - Overlap Treatment: Treatment D (10-Day Cooldown)")
    print("  - Execution: Buy SPY at Close of Spike Day, Exit after 3 Days")
    print("====================================================\n")
    
    # 1. Load Data
    db_path = project_root / "data" / "database" / "market_data.db"
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return
        
    conn = sqlite3.connect(str(db_path))
    # Load recent data (from 2023 to present)
    vix_df = pd.read_sql('SELECT * FROM "^VIX" WHERE INTERVAL = "1d"', conn)
    spy_df = pd.read_sql('SELECT * FROM "SPY" WHERE INTERVAL = "1d"', conn)
    conn.close()
    
    vix_df['DATE'] = pd.to_datetime(vix_df['DATE'])
    spy_df['DATE'] = pd.to_datetime(spy_df['DATE'])
    
    df = pd.merge(vix_df[['DATE', 'CLOSE']], spy_df[['DATE', 'CLOSE']], on='DATE', suffixes=('_VIX', '_SPY'), how='inner')
    df = df.rename(columns={'CLOSE_VIX': 'CLOSE', 'CLOSE_SPY': 'CLOSE_SPY'})
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # 2. Generate Signals
    df = compute_signals(df)
    
    # Apply optimal filter (Z3 and Treatment D)
    sig_col = 'Sig_Lvl_Z3'
    df['Filtered_Signal'] = filter_signal_overlap(df, sig_col, horizon=3, treatment='D', cooldown_days=10)
    
    # Filter for recent period (e.g., from 2023-01-01 onwards) for clear demonstration
    demo_df = df[df['DATE'] >= '2023-01-01'].reset_index(drop=True)
    
    trade_signals = demo_df[demo_df['Filtered_Signal'] == True]
    print(f"Found {len(trade_signals)} trading events since 2023-01-01:\n")
    
    trades = []
    initial_capital = 10000.0
    capital = initial_capital
    
    print("| Trade # | Entry Date | VIX Level | SPY Entry | Exit Date | SPY Exit | Trade Return | Portfolio Value |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for idx, row in enumerate(trade_signals.itertuples()):
        entry_idx = row.Index
        entry_date = row.DATE
        vix_lvl = row.CLOSE
        spy_entry_price = row.CLOSE_SPY
        
        # Exit index is entry_idx + 3 days
        exit_idx = min(entry_idx + 3, len(demo_df) - 1)
        exit_row = demo_df.iloc[exit_idx]
        exit_date = exit_row.DATE
        spy_exit_price = exit_row.CLOSE_SPY
        
        # Calculate return (with 0.05% slippage/commission on each trade leg)
        slippage = 0.0005
        net_ret = (spy_exit_price / spy_entry_price) * (1.0 - slippage) * (1.0 - slippage) - 1.0
        
        capital *= (1.0 + net_ret)
        
        print(f"| {idx+1:02d} | {entry_date.strftime('%Y-%m-%d')} | {vix_lvl:.2f} | ${spy_entry_price:.2f} | "
              f"{exit_date.strftime('%Y-%m-%d')} | ${spy_exit_price:.2f} | {net_ret:+.2%} | ${capital:,.2f} |")
              
        trades.append({
            'trade_num': idx + 1,
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'vix': vix_lvl,
            'spy_entry': spy_entry_price,
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'spy_exit': spy_exit_price,
            'return': net_ret
        })
        
    print("\n====================================================")
    total_return = (capital / initial_capital) - 1.0
    print(f"Demo Summary:")
    print(f"  - Initial Capital: ${initial_capital:,.2f}")
    print(f"  - Final Capital  : ${capital:,.2f}")
    print(f"  - Total Return   : {total_return:+.2%}")
    print("====================================================")

if __name__ == "__main__":
    main()
