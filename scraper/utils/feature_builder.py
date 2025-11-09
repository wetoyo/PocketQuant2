# scraper/utils/features/feature_builder.py
from pathlib import Path
import pandas as pd
from ..features import compute_returns, compute_moving_averages, compute_bollinger_bands, compute_rsi, compute_macd, compute_volatility, compute_atr


FEATURES_FOLDER = Path("data/features")
FEATURES_FOLDER.mkdir(parents=True, exist_ok=True)

def build_features(
    df: pd.DataFrame,
    save_path: Path = None,
    returns=True,
    log_returns=True,
    ma_windows=[5,20,50],
    bb_windows=[20],
    rsi_windows=[14],
    macd_params=[(12,26,9)],
    vol_windows=[10],
    atr_windows=[14]
):
    save_path = save_path or FEATURES_FOLDER
    save_path.mkdir(parents=True, exist_ok=True)

    features = []

    if returns:
        features.append(compute_returns(df, log=False))
    if log_returns:
        features.append(compute_returns(df, log=True))
    if ma_windows:
        features.append(compute_moving_averages(df, ma_windows))
    if bb_windows:
        features.append(compute_bollinger_bands(df, bb_windows))
    if rsi_windows:
        features.append(compute_rsi(df, rsi_windows))
    if macd_params:
        features.append(compute_macd(df, macd_params))
    if vol_windows:
        features.append(compute_volatility(df, vol_windows))
    if atr_windows:
        features.append(compute_atr(df, atr_windows))

    # Merge all features on DATE
    from functools import reduce
    df_features = reduce(lambda left,right: pd.merge(left,right,on="DATE", how="outer"), features)

    return df_features
