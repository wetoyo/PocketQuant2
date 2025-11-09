import pandas as pd
def compute_atr(df, windows=[14]):
    """
    Compute Average True Range (ATR) for given windows.
    """
    result = df[["DATE"]].copy()

    high_low = df["HIGH"] - df["LOW"]
    high_close = (df["HIGH"] - df["CLOSE"].shift(1)).abs()
    low_close = (df["LOW"] - df["CLOSE"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    for w in windows:
        result[f"ATR_{w}"] = tr.rolling(w).mean()

    return result
