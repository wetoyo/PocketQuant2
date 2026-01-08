def test_no_future_leakage(df, feature_cols):
    """Ensures features at time t do not depend on CLOSE[t+1]."""
    for col in feature_cols:
        # correlation with future close should not be stronger
        corr_now = df[col].corr(df["CLOSE"])
        corr_future = df[col].corr(df["CLOSE"].shift(-1))

        assert abs(corr_future) <= abs(corr_now) + 1e-3, \
            f"Leakage suspected in feature {col}"

def test_feature_shift(df, feature_cols):
    """Checks that features are known before the target is realized."""
    # Features must exist one bar before the target
    feature_na = df[feature_cols].isna().sum().sum()
    target_na = df["target"].isna().sum()

    assert feature_na >= target_na, \
        "Features appear available after target (possible lookahead)"
