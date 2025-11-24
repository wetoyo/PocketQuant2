import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

root = Path(__file__).parent.parent  # project_root
sys.path.append(str(root))
from scraper.setup import BaseSetup

def prepare_data(df):
    """
    Prepare data for XGBoost model.
    """
    # Create target: 1 if next day's close is higher than today's close, else 0
    df['Target'] = (df['CLOSE'].shift(-1) > df['CLOSE']).astype(int)
    
    # Drop the last row as it will have NaN target
    df = df.dropna()
    
    return df

def train_model(df, features):
    """
    Train XGBoost model.
    """
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def main():
    # Initialize setup with desired features
    setup = BaseSetup(
        tickers=["AAPL", "MSFT"],
        scraper_type="yfinance",
        start_date="2020-01-01",
        end_date="2023-01-01",
        interval="1d",
        adjusted=True,
        fill_missing=True,
        features_options={
            "returns": True,
            "log_returns": True,
            "ma_windows": [5, 20, 50],
            "bb_windows": [20],
            "rsi_windows": [14],
            "macd_params": [(12, 26, 9)],
            "vol_windows": [10],
            "atr_windows": [14]
        }
    )
    
    # Run pipeline to get data and features
    features_dict = setup.run_pipeline()
    
    # Combine features for all tickers or train per ticker
    # For this example, let's train on AAPL
    ticker = "AAPL"
    if ticker in features_dict:
        print(f"\nTraining model for {ticker}...")
        df_features = features_dict[ticker]
        
        # Merge CLOSE from raw data to calculate Target
        if ticker in setup.data:
            df_raw = setup.data[ticker]
            # Ensure DATE is consistent format if needed, but usually it is
            if 'CLOSE' in df_raw.columns:
                df = pd.merge(df_features, df_raw[['DATE', 'CLOSE']], on='DATE', how='inner')
            else:
                print(f"CLOSE column missing in raw data for {ticker}")
                return
        else:
            print(f"Raw data missing for {ticker}")
            return
        
        # Ensure features exist
        required_features = ['MA_5', 'MA_20', 'MA_50', 'RSI_14']
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            print(f"Missing features: {missing_features}")
            return

        df_prepared = prepare_data(df)
        
        model = train_model(df_prepared, required_features)
        
        # Feature Importance
        print("\nFeature Importance:")
        importance = pd.DataFrame({
            'Feature': required_features,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print(importance)
    else:
        print(f"No data found for {ticker}")

if __name__ == "__main__":
    main()
