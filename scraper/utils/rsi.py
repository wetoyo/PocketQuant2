def compute_rsi(df, windows=[14]):
    """
    Compute Relative Strength Index (RSI) for given windows.
    """
    import pandas as pd
    result = df[["DATE"]].copy()

    for w in windows:
        delta = df["CLOSE"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ema_up = up.ewm(span=w, adjust=False).mean()
        ema_down = down.ewm(span=w, adjust=False).mean()
        rs = ema_up / ema_down
        result[f"RSI_{w}"] = 100 - (100 / (1 + rs))

    return result
