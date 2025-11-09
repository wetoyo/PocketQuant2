import pandas as pd
import numpy as np

def compute_returns(df: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """
    Compute simple or log returns.
    """
    result = df[["DATE"]].copy()
    if log:
        result["LOG_RETURN"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
    else:
        result["RETURN"] = df["CLOSE"].pct_change()
    return result
