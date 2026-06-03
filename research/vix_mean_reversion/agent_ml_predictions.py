import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pathlib import Path
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(current_dir))

from analyzer import compute_signals

def engineer_features_local(df: pd.DataFrame) -> tuple:
    """
    Creates rolling and lag-free features for the ML model.
    """
    df = df.copy()
    
    # 1. SPY rolling features
    df['SPY_RET'] = df['CLOSE_SPY'].pct_change()
    df['SPY_VOL_20'] = df['SPY_RET'].rolling(20).std() * np.sqrt(252)
    df['SPY_MA_200'] = df['CLOSE_SPY'].rolling(200).mean()
    df['SPY_TREND'] = df['CLOSE_SPY'] / df['SPY_MA_200'] - 1.0
    
    spy_roll_max = df['CLOSE_SPY'].rolling(252, min_periods=30).max()
    df['SPY_DD'] = (df['CLOSE_SPY'] - spy_roll_max) / spy_roll_max
    
    # Fill any missing values
    df['SPY_VOL_20'] = df['SPY_VOL_20'].ffill().bfill()
    df['SPY_TREND'] = df['SPY_TREND'].ffill().bfill()
    df['SPY_DD'] = df['SPY_DD'].ffill().bfill()
    
    # 2. VIX features
    df['VIX_RET_5'] = df['CLOSE'].pct_change(5)
    df['VIX_VOL_20'] = df['VIX_RET'].rolling(20).std() * np.sqrt(252)
    
    # Rename for clarity
    df['VIX_LVL'] = df['CLOSE']
    df['VIX_Z'] = df['VIX_LEVEL_Z']
    
    feature_cols = [
        'VIX_LVL', 'VIX_Z', 'VIX_RET_Z', 'VIX_RET_5', 'VIX_VOL_20', 
        'SPY_VOL_20', 'SPY_TREND', 'SPY_DD'
    ]
    
    # Target columns
    for h in [5, 10, 20]:
        df[f'target_ret_{h}'] = df['CLOSE'].shift(-h) / df['CLOSE'] - 1.0
        
    fwd_closes = []
    for h in range(1, 21):
        fwd_closes.append(df['CLOSE'].shift(-h))
        
    fwd_matrix = np.column_stack(fwd_closes)
    df['target_opt_day'] = np.argmin(fwd_matrix, axis=1) + 1
    
    return df, feature_cols

def main():
    print("Agent ML Predictions: Starting...")
    
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
    
    # 2. Compute Signals and Engineer Features
    df = compute_signals(df)
    df, feature_cols = engineer_features_local(df)
    
    rep_sig = 'Sig_Lvl_Z2'
    target_col = 'target_ret_10'
    train_end_date = '2015-12-31'
    
    # Keep only event days where target is not NaN
    event_df = df[df[rep_sig] == True].copy()
    event_df = event_df.dropna(subset=[target_col] + feature_cols)
    
    if len(event_df) < 50:
        print("Error: Too few events for ML training.")
        return
        
    # Split Train/Test
    train_mask = event_df['DATE'] <= pd.to_datetime(train_end_date)
    test_mask = ~train_mask
    
    X_train = event_df.loc[train_mask, feature_cols]
    y_train = event_df.loc[train_mask, target_col]
    X_test = event_df.loc[test_mask, feature_cols]
    y_test = event_df.loc[test_mask, target_col]
    test_dates = event_df.loc[test_mask, 'DATE']
    
    # Train Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_r2 = float(ridge.score(X_test, y_test))
    ridge_mae = float(np.mean(np.abs(ridge_preds - y_test)))
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_r2 = float(rf.score(X_test, y_test))
    rf_mae = float(np.mean(np.abs(rf_preds - y_test)))
    
    # Train XGBoost
    xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_r2 = float(xgb.score(X_test, y_test))
    xgb_mae = float(np.mean(np.abs(xgb_preds - y_test)))
    
    metrics = {
        'train_count': int(len(X_train)),
        'test_count': int(len(X_test)),
        'ridge_r2': ridge_r2,
        'ridge_mae': ridge_mae,
        'rf_r2': rf_r2,
        'rf_mae': rf_mae,
        'xgb_r2': xgb_r2,
        'xgb_mae': xgb_mae,
        'feature_importances': {f: float(imp) for f, imp in zip(feature_cols, rf.feature_importances_)}
    }
    
    # Generate scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(xgb_preds, y_test, alpha=0.6, color='blue', edgecolor='k')
    m, b = np.polyfit(xgb_preds, y_test, 1)
    plt.plot(xgb_preds, m*xgb_preds + b, color='red', linewidth=2, label=f"Fit (Slope: {m:.2f})")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.title("XGBoost Predictions vs. Actual VIX 10d Forward Returns (OOS Events)", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted 10d Return", fontsize=12)
    plt.ylabel("Actual 10d Return", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plots_dir = current_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "ml_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save Results JSON
    output_dir = current_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ml_predictions.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Agent ML Predictions: Completed successfully.")

if __name__ == "__main__":
    main()
