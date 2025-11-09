def compute_volatility(df, windows=[10]):
    """
    Compute rolling standard deviation of returns as volatility.
    """
    result = df[["DATE"]].copy()
    returns = df["CLOSE"].pct_change()

    for w in windows:
        result[f"VOL_{w}"] = returns.rolling(w).std()

    return result
