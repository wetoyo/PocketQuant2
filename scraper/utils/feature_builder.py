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
    ma_windows=None,
    bb_windows=None,
    rsi_windows=None,
    macd_params=None,
    vol_windows=None,
    atr_windows=None
):
    save_path = save_path or FEATURES_FOLDER
    ma_windows = ma_windows or [5, 20, 50]
    bb_windows = bb_windows or [20]
    rsi_windows = rsi_windows or [14]
    macd_params = macd_params or [(12, 26, 9)]
    vol_windows = vol_windows or [10]
    atr_windows = atr_windows or [14]
    save_path.mkdir(parents=True, exist_ok=True)
    ticker = None
    if isinstance(df.columns, pd.MultiIndex):
        ticker = df.columns.get_level_values(1)[1]
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
    for i, f in enumerate(features):
        if isinstance(f.columns, pd.MultiIndex):
            f.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in f.columns]
            features[i] = f
    df_features = reduce(lambda left,right: pd.merge(left,right,on="DATE", how="outer"), features)
    if save_path:
        filename = f"{ticker}_features.csv" if ticker else "features.csv"
        df_features.to_csv(save_path / filename, index=False)


    return df_features
