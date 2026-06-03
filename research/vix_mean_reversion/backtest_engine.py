import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

def generate_signals(signals: pd.Series, H: int, treatment: str, cooldown_days: int = 10) -> Tuple[pd.Series, pd.Series]:
    """
    Generates entry and exit boolean Series from raw signals based on holding period H and treatment.
    
    Treatments:
      - 'A': Keep all signals (note: parallel trades are handled separately in portfolio modeling,
             but for from_signals, we generate entries/exits with exits shifted by H days).
      - 'B': Ignore new signals while a trade is active (holding period H).
      - 'C': Only first signal in a consecutive cluster of signal days, holding period H.
      - 'D': Cooldown period of N days after a signal, holding period H.
    """
    entries = pd.Series(False, index=signals.index)
    exits = pd.Series(False, index=signals.index)
    n = len(signals)
    
    if treatment == 'B':
        in_trade = False
        entry_idx = -1
        for i in range(n):
            if in_trade:
                if i - entry_idx >= H:
                    exits.iloc[i] = True
                    in_trade = False
                    # Allow immediate re-entry if there is a signal on the exit day
                    if signals.iloc[i]:
                        entries.iloc[i] = True
                        entry_idx = i
                        in_trade = True
            else:
                if signals.iloc[i]:
                    entries.iloc[i] = True
                    entry_idx = i
                    in_trade = True
        if in_trade:
            exits.iloc[-1] = True
            
    elif treatment == 'C':
        # First filter signals for cluster start
        cluster_signals = []
        for i in range(n):
            if signals.iloc[i]:
                if i == 0 or not signals.iloc[i-1]:
                    cluster_signals.append(True)
                else:
                    cluster_signals.append(False)
            else:
                cluster_signals.append(False)
        cluster_signals = pd.Series(cluster_signals, index=signals.index)
        
        # Apply Treatment B logic to cluster starts
        in_trade = False
        entry_idx = -1
        for i in range(n):
            if in_trade:
                if i - entry_idx >= H:
                    exits.iloc[i] = True
                    in_trade = False
                    if cluster_signals.iloc[i]:
                        entries.iloc[i] = True
                        entry_idx = i
                        in_trade = True
            else:
                if cluster_signals.iloc[i]:
                    entries.iloc[i] = True
                    entry_idx = i
                    in_trade = True
        if in_trade:
            exits.iloc[-1] = True
            
    elif treatment == 'D':
        # Cooldown of cooldown_days
        in_trade = False
        entry_idx = -1
        for i in range(n):
            if in_trade:
                if i - entry_idx >= H:
                    exits.iloc[i] = True
                    in_trade = False
                    # Check if cooldown is over
                    if i - entry_idx >= cooldown_days and signals.iloc[i]:
                        entries.iloc[i] = True
                        entry_idx = i
                        in_trade = True
            else:
                if signals.iloc[i]:
                    entries.iloc[i] = True
                    entry_idx = i
                    in_trade = True
        if in_trade:
            exits.iloc[-1] = True
            
    else:  # Treatment 'A' (parallel trades)
        # Every signal triggers a trade. For vectorbt, multiple parallel trades are represented
        # by running separate columns or by manual return accumulation.
        # For simplicity in from_signals, we just exit H days after every entry.
        # (This will result in overlapping entries/exits which standard long-only vectorbt handles by accumulation).
        for i in range(n):
            if signals.iloc[i]:
                entries.iloc[i] = True
                exit_idx = min(i + H, n - 1)
                exits.iloc[exit_idx] = True
                
    return entries, exits

def calculate_portfolio_metrics(equity: pd.Series, cash_returns: pd.Series = None) -> Dict:
    """
    Computes standard portfolio metrics from an equity curve Series.
    """
    if len(equity) < 2:
        return {}
        
    returns = equity.pct_change().dropna()
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    
    # Calculate CAGR
    n_days = len(equity)
    years = n_days / 252.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    
    # Volatility (annualized)
    vol = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (assume 0 risk-free rate)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan
    
    # Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 1 and downside_returns.std() > 0 else np.nan
    
    # Max Drawdown
    cum_max = equity.cummax()
    drawdowns = (equity - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    
    # Win rate of daily returns
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_dd,
        'Volatility': vol,
        'Win Rate': win_rate
    }

def run_idealized_backtest(
    df: pd.DataFrame, 
    signal_col: str, 
    H: int, 
    treatment: str, 
    asset_col: str = 'CLOSE', # 'CLOSE' for VIX, or SPY prices
    is_short: bool = True,
    transaction_cost: float = 0.001, # 0.1% per trade
    borrow_rate_annual: float = 0.05, # 5% annual borrowing cost for shorting
    cooldown_days: int = 10
) -> Tuple[pd.Series, pd.Series]:
    """
    Runs a realistic backtest accounting for transaction costs and borrowing fees.
    Returns:
      equity: pd.Series of the equity curve (starting at 1.0)
      positions: pd.Series indicating active position size (-1 for short, 1 for long, 0 for cash)
    """
    prices = df[asset_col].values
    n = len(prices)
    
    # Generate entries and exits
    signals = df[signal_col]
    entries, exits = generate_signals(signals, H, treatment, cooldown_days)
    
    # Track positions and cash
    equity = np.ones(n)
    positions = np.zeros(n) # Active position flags (-1, 0, 1)
    
    # If Treatment A (parallel trades), we can have multiple active units.
    # To handle this cleanly and realistically:
    # We allocate equal weight to each active trade.
    # Let's track active trades as a list of dicts: [{'entry_idx': i, 'exit_idx': j, 'entry_price': p}]
    active_trades = []
    
    for i in range(n):
        # Update equity from yesterday
        if i > 0:
            equity[i] = equity[i-1]
            
            if len(active_trades) > 0:
                # Calculate daily return from active positions
                # For short positions: daily return is -1 * price return
                # For long positions: daily return is 1 * price return
                price_ret = (prices[i] / prices[i-1]) - 1.0
                
                # Daily borrowing cost
                daily_borrow = borrow_rate_annual / 252.0
                
                # Portfolio return is the average of active trade returns
                # (Assuming we divide our capital equally among active trades, with the rest in cash)
                trade_returns = []
                for trade in active_trades:
                    direction = -1.0 if is_short else 1.0
                    ret = direction * price_ret - (daily_borrow if is_short else 0.0)
                    trade_returns.append(ret)
                
                # We assume we allocate 100% of capital divided equally among active trades
                avg_ret = np.mean(trade_returns)
                equity[i] = equity[i-1] * (1.0 + avg_ret)
                
        # Check exits
        if len(active_trades) > 0:
            # Filter out trades that have reached their exit index
            still_active = []
            for trade in active_trades:
                if i >= trade['exit_idx']:
                    # Apply transaction cost on exit
                    equity[i] *= (1.0 - transaction_cost)
                else:
                    still_active.append(trade)
            active_trades = still_active
            
        # Check entries
        if entries.iloc[i]:
            # Trigger a new trade
            # In Treatment B, C, D we can have at most 1 active trade.
            # In Treatment A we can have multiple.
            if treatment in ['B', 'C', 'D'] and len(active_trades) > 0:
                # Should not happen due to signal filtering, but just in case
                pass
            else:
                exit_idx = min(i + H, n - 1)
                active_trades.append({
                    'entry_idx': i,
                    'exit_idx': exit_idx,
                    'entry_price': prices[i]
                })
                # Apply transaction cost on entry
                equity[i] *= (1.0 - transaction_cost)
                
        # Record position size for reporting
        if len(active_trades) > 0:
            positions[i] = -1.0 if is_short else 1.0
            
    equity_series = pd.Series(equity, index=df['DATE'])
    position_series = pd.Series(positions, index=df['DATE'])
    return equity_series, position_series

def walk_forward_optimization(
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    signal_name_prefix: str, # e.g. 'Sig_Lvl_Z'
    is_short: bool = True,
    train_start: str = '1995-01-01',
    train_end: str = '2015-12-31',
    val_start: str = '2016-01-01',
    val_end: str = '2020-12-31',
    test_start: str = '2021-01-01',
    test_end: str = '2026-06-01',
    asset_col: str = 'CLOSE'
) -> Dict:
    """
    Implements a fixed train-validation-test walk-forward validation.
    Optimizes z-score threshold (2.0 vs 3.0) and holding period H (5, 10, 20) on Train,
    evaluates on Val, and tests on Test.
    """
    df = df.copy()
    df['DATE_STR'] = df['DATE'].dt.strftime('%Y-%m-%d')
    
    # Parameter grid
    thresholds = [2.0, 3.0]
    holding_periods = [3, 5, 10, 20]
    treatments = ['B', 'C', 'D']
    
    # Train set
    train_mask = (df['DATE_STR'] >= train_start) & (df['DATE_STR'] <= train_end)
    train_df = df[train_mask].reset_index(drop=True)
    
    best_sharpe = -999.0
    best_params = None
    
    print(f"Optimizing parameters on Train set ({train_start} to {train_end})...")
    for thresh in thresholds:
        # Use corresponding signal column
        sig_col = f'Sig_Lvl_Z{int(thresh)}' if signal_name_prefix == 'Sig_Lvl_Z' else f'Sig_Ret_Z{int(thresh)}'
        if sig_col not in train_df.columns:
            continue
            
        for H in holding_periods:
            for treat in treatments:
                equity, _ = run_idealized_backtest(
                    train_df, sig_col, H, treat, asset_col=asset_col, is_short=is_short
                )
                metrics = calculate_portfolio_metrics(equity)
                sharpe = metrics.get('Sharpe', -999.0)
                
                # We want a positive CAGR and max Sharpe
                if not np.isnan(sharpe) and sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = {'threshold': thresh, 'H': H, 'treatment': treat}
                    
    print(f"Best parameters found: {best_params} with Train Sharpe: {best_sharpe:.4f}")
    
    if best_params is None:
        # Default fallback
        best_params = {'threshold': 2.0, 'H': 5, 'treatment': 'B'}
        
    # Evaluate on Validation set
    val_mask = (df['DATE_STR'] >= val_start) & (df['DATE_STR'] <= val_end)
    val_df = df[val_mask].reset_index(drop=True)
    sig_col_opt = f'Sig_Lvl_Z{int(best_params["threshold"])}' if signal_name_prefix == 'Sig_Lvl_Z' else f'Sig_Ret_Z{int(best_params["threshold"])}'
    
    val_equity, _ = run_idealized_backtest(
        val_df, sig_col_opt, best_params['H'], best_params['treatment'], asset_col=asset_col, is_short=is_short
    )
    val_metrics = calculate_portfolio_metrics(val_equity)
    
    # Evaluate on Test set (Out-of-Sample)
    test_mask = (df['DATE_STR'] >= test_start) & (df['DATE_STR'] <= test_end)
    test_df = df[test_mask].reset_index(drop=True)
    
    test_equity, test_pos = run_idealized_backtest(
        test_df, sig_col_opt, best_params['H'], best_params['treatment'], asset_col=asset_col, is_short=is_short
    )
    test_metrics = calculate_portfolio_metrics(test_equity)
    
    return {
        'best_params': best_params,
        'train_sharpe': best_sharpe,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_equity': test_equity,
        'test_positions': test_pos
    }
