import pytest
import pandas as pd
import numpy as np
from models.daily_model.xgboost.xgboost_model import model

@pytest.fixture(scope="session")
def model_results():
    """Run the model once on a fixed small dataset."""
    ticker = "AAPL"
    # Fixed small dataset for consistency
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    results = model(ticker, start_date, end_date)
    return results

@pytest.fixture
def df(model_results):
    return model_results["df"]

@pytest.fixture
def feature_cols(model_results):
    return model_results["feature_names"]

@pytest.fixture
def X_train(model_results):
    df = model_results["df"]
    feature_cols = model_results["feature_names"]
    split = int(len(df) * 0.8)
    return df[feature_cols].iloc[:split]

@pytest.fixture
def X_test(model_results):
    return model_results["X_test"]

@pytest.fixture
def y_test(model_results):
    return model_results["y_test"]

@pytest.fixture
def probs_test(model_results):
    return model_results["probs_test"]
