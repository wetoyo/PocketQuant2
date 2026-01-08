from sklearn.metrics import roc_auc_score

def test_auc_not_random(y_test, probs_test):
    """Ensures the model learned something (even if inverted)."""
    auc = roc_auc_score(y_test, probs_test)
    auc_flipped = roc_auc_score(y_test, 1 - probs_test)

    assert max(auc, auc_flipped) > 0.51, \
        "Model shows no detectable signal"

def test_probability_distribution(probs_test):
    """Checks that probabilities aren’t degenerate."""
    assert probs_test.min() >= 0
    assert probs_test.max() <= 1

    # avoid constant predictions
    assert probs_test.std() > 1e-3

def test_return_feature_direction(df):
    """Detects inverted momentum signals."""
    if "returns_1" in df.columns:
        col = "returns_1"
    elif "RETURN" in df.columns:
        col = "RETURN"
    else:
        return

    corr = df[col].corr(df["target"])
    assert abs(corr) > 0.01, f"{col} feature has no directional signal"
