def compute_macd(df, params=[(12,26,9)]):
    """
    Compute MACD and Signal line.
    params: list of tuples (fast, slow, signal)
    """
    result = df[["DATE"]].copy()

    for fast, slow, signal in params:
        ema_fast = df["CLOSE"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["CLOSE"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        result[f"MACD_{fast}_{slow}_{signal}"] = macd_line
        result[f"MACD_SIGNAL_{fast}_{slow}_{signal}"] = signal_line

    return result
