def compute_bollinger_bands(df, windows=[20]):
    """
    Compute Bollinger Bands (upper and lower) for given windows.
    """
    result = df[["DATE"]].copy()
    for w in windows:
        ma = df["CLOSE"].rolling(w).mean()
        std = df["CLOSE"].rolling(w).std()
        result[f"BB_UPPER_{w}"] = ma + 2*std
        result[f"BB_LOWER_{w}"] = ma - 2*std
    return result
