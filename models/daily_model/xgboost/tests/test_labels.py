import numpy as np

def test_target_definition(df):
    """Verifies that target == 1 really means next bar close is higher."""
    # pick random rows away from the end
    idx = np.random.randint(0, len(df) - 5)

    close_t = df.iloc[idx]["CLOSE"]
    close_t1 = df.iloc[idx + 1]["CLOSE"]
    target = df.iloc[idx]["target"]

    assert target == int(close_t1 > close_t)
