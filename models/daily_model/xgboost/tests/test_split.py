def test_time_split_order(X_train, X_test):
    """Ensures your split is strictly chronological."""
    last_train_index = X_train.index.max()
    first_test_index = X_test.index.min()

    assert last_train_index < first_test_index
