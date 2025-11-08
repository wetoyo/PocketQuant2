# scraper/utils/features/moving_averages.py
def compute_moving_averages(df, windows=[5,20,50]):
    """
    Compute multiple moving averages.
    Returns a DataFrame with DATE + MA columns.
    """
    result = df[["DATE"]].copy()
    for w in windows:
        result[f"MA_{w}"] = df["CLOSE"].rolling(w).mean()
    return result
