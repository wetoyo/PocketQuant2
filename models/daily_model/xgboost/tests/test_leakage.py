import numpy as np
from scraper.utils.feature_builder import build_features

# Must match the features_options used in xgboost_model.model()
FEATURES_OPTIONS = {
    "returns": True,
    "ma_windows": [5, 20, 50],
    "rsi_windows": [14],
    "vol_windows": [10],
}

def test_no_future_leakage(df, feature_cols, tmp_path):
    """Ensures features at time t do not depend on any data after t-1.

    Recomputes the features after scrambling every bar after a cutoff k. If the
    features only use past data, rows up to k+1 (features are shifted by one bar)
    must be identical to the unperturbed computation.
    """
    price_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    raw = df[["DATE"] + price_cols].reset_index(drop=True)
    raw[price_cols] = raw[price_cols].astype(float)
    base = build_features(raw, save_path=tmp_path, **FEATURES_OPTIONS)
    missing = set(feature_cols) - set(base.columns)
    assert not missing, f"Features not produced by build_features: {missing}"

    rng = np.random.default_rng(0)
    for k in [len(raw) // 4, len(raw) // 2, 3 * len(raw) // 4]:
        perturbed = raw.copy()
        future = perturbed.index > k
        for col in price_cols:
            perturbed.loc[future, col] *= rng.uniform(0.5, 1.5, future.sum())
        scrambled = build_features(perturbed, save_path=tmp_path, **FEATURES_OPTIONS)

        for col in feature_cols:
            assert np.allclose(
                base.loc[:k + 1, col], scrambled.loc[:k + 1, col], equal_nan=True
            ), f"Leakage suspected in feature {col} (changed by data after row {k})"

def test_feature_shift(df, feature_cols):
    """Checks that features are known before the target is realized."""
    # Features must exist one bar before the target
    feature_na = df[feature_cols].isna().sum().sum()
    target_na = df["target"].isna().sum()

    assert feature_na >= target_na, \
        "Features appear available after target (possible lookahead)"
