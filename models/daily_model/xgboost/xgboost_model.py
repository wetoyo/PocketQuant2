import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from scraper.setup import BaseSetup

def model(ticker, start_date, end_date, interval="1d"):
    setup = BaseSetup(
            tickers=[ticker],
            scraper_type="yfinance",
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            features_options={
                "returns": True,
                "ma_windows": [5, 20, 50],
                "rsi_windows": [14],
                "vol_windows": [10],
            }
        )
    
    setup.run_pipeline()
    df_features = setup.features[ticker]
    df_raw = setup.data_dict[ticker]
    
    # Merge raw data with features
    df = pd.merge(df_raw, df_features, on="DATE", how="inner")
    
    df["target"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)
    df = df.dropna()
    feature_cols = [
        c for c in df.columns
        if c not in ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "target", "TICKER", "INTERVAL"]
    ]

    X = df[feature_cols].shift(1)
    y = df["target"]

    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    probs_train = model.predict_proba(X_train)[:, 1]
    probs_test  = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, probs_train)
    auc_test  = roc_auc_score(y_test, probs_test)
    auc_flip = roc_auc_score(y_test, 1 - probs_test)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "probs_test": probs_test,
        "probs_train": probs_train,
        "auc_train": auc_train,
        "auc_test": auc_test,
        "feature_names": feature_cols,
        "df": df,
    }

if __name__ == "__main__":
    from datetime import datetime, timedelta

    ticker = "AAPL"
    
    # Use last 8 days for quick test
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)

    print(f"Testing XGBoost model for {ticker} (last 60 days, 1-minute bars)...")

    results = model(
        ticker=ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1m"  # 1-minute bars
    )

    print("\nResults Summary:")
    print(f"AUC Train: {results['auc_train']:.4f}")
    print(f"AUC Test:  {results['auc_test']:.4f}")
    print(f"Features:  {len(results['feature_names'])}")