import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import List, Dict, Tuple, Optional

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Creates predictive features for the ML model.
    All features are rolling and lag-free to prevent lookahead bias.
    """
    df = df.copy()
    
    # 1. Calculate SPY rolling features
    df['SPY_RET'] = df['CLOSE_SPY'].pct_change()
    df['SPY_VOL_20'] = df['SPY_RET'].rolling(20).std() * np.sqrt(252)
    df['SPY_MA_200'] = df['CLOSE_SPY'].rolling(200).mean()
    df['SPY_TREND'] = df['CLOSE_SPY'] / df['SPY_MA_200'] - 1.0
    
    # SPY drawdown
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
    
    # Features list
    feature_cols = [
        'VIX_LVL', 'VIX_Z', 'VIX_RET_Z', 'VIX_RET_5', 'VIX_VOL_20', 
        'SPY_VOL_20', 'SPY_TREND', 'SPY_DD'
    ]
    
    # Add target columns for multi-horizon returns
    for h in [5, 10, 20]:
        df[f'target_ret_{h}'] = df['CLOSE'].shift(-h) / df['CLOSE'] - 1.0
        
    # Add target for optimal holding period: day in 1..20 where VIX reaches lowest close
    fwd_closes = []
    for h in range(1, 21):
        fwd_closes.append(df['CLOSE'].shift(-h))
        
    fwd_matrix = np.column_stack(fwd_closes) # Shape (n_rows, 20)
    df['target_opt_day'] = np.argmin(fwd_matrix, axis=1) + 1
    
    return df, feature_cols

def train_predict_ml(
    df: pd.DataFrame,
    feature_cols: List[str],
    signal_col: str,
    target_col: str = 'target_ret_10',
    train_end_date: str = '2015-12-31'
) -> Tuple[Dict, pd.Series, pd.Series]:
    """
    Trains Ridge, Random Forest, and XGBoost regressor models on volatility shock events.
    Splits data chronologically.
    Returns model metrics and test predictions.
    """
    # Keep only event days
    event_df = df[df[signal_col] == True].copy()
    
    # Drop rows where target is NaN (at the end of dataset)
    event_df = event_df.dropna(subset=[target_col] + feature_cols)
    
    if len(event_df) < 50:
        return {'Ridge_R2': np.nan, 'RF_R2': np.nan, 'XGB_R2': np.nan}, pd.Series(), pd.Series()
        
    # Split Train/Test chronologically
    train_mask = event_df['DATE'] <= pd.to_datetime(train_end_date)
    test_mask = ~train_mask
    
    X_train = event_df.loc[train_mask, feature_cols]
    y_train = event_df.loc[train_mask, target_col]
    X_test = event_df.loc[test_mask, feature_cols]
    y_test = event_df.loc[test_mask, target_col]
    test_dates = event_df.loc[test_mask, 'DATE']
    
    if len(X_train) < 20 or len(X_test) < 10:
        return {'Ridge_R2': np.nan, 'RF_R2': np.nan, 'XGB_R2': np.nan}, pd.Series(), pd.Series()
        
    # Train Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_r2 = ridge.score(X_test, y_test)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_r2 = rf.score(X_test, y_test)
    
    # Train XGBoost
    xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_r2 = xgb.score(X_test, y_test)
    
    # Calculate Mean Absolute Error (MAE)
    ridge_mae = np.mean(np.abs(ridge_preds - y_test))
    rf_mae = np.mean(np.abs(rf_preds - y_test))
    xgb_mae = np.mean(np.abs(xgb_preds - y_test))
    
    metrics = {
        'Train_Count': len(X_train),
        'Test_Count': len(X_test),
        'Ridge_R2': ridge_r2,
        'Ridge_MAE': ridge_mae,
        'RF_R2': rf_r2,
        'RF_MAE': rf_mae,
        'XGB_R2': xgb_r2,
        'XGB_MAE': xgb_mae,
        'Feature_Importances': dict(zip(feature_cols, rf.feature_importances_))
    }
    
    return metrics, pd.Series(xgb_preds, index=test_dates), pd.Series(y_test.values, index=test_dates)
