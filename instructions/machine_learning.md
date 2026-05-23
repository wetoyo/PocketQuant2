# Machine Learning Framework (XGBoost)

This document describes the machine learning infrastructure of **PocketQuant2**, focusing on the XGBoost classification models.

---

## 1. Model Pipeline Architecture

The machine learning implementation resides in [models/price_prediction_xgb.py](file:///d:/Files/Code/PocketQuant2/models/price_prediction_xgb.py) and [models/daily_model/xgboost/xgboost_model.py](file:///d:/Files/Code/PocketQuant2/models/daily_model/xgboost/xgboost_model.py).

The modeling pipeline consists of:
1.  **Pipeline Ingestion**: `BaseSetup` retrieves raw data and builds features (returns, moving averages, rsi, volatility).
2.  **Dataset Merging**: Raw price data is merged with calculated features on `DATE`.
3.  **Target Labeling**: Defining the binary price direction target.
4.  **Leakage Prevention**: Shifting features to align with the target date.
5.  **Chronological Split**: Splitting the dataset into train and test sets without shuffling.
6.  **XGBoost Classifier Training**: Fitting the model and predicting out-of-sample probabilities.
7.  **Evaluation**: Scoring train/test ROC AUC and checking feature importances.

---

## 2. Target Labeling & Feature Matrix Construction

### A. Target Variable Definition
The goal of the classifier is to predict whether the close of the *next* bar ($t+1$) will be higher than the close of the *current* bar ($t$). The target is defined as:
```python
df["target"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)
```
*   `target = 1` if $C_{t+1} > C_t$.
*   `target = 0` if $C_{t+1} \leq C_t$.
*   The last row in the DataFrame is dropped because its target is `NaN` (no $t+1$ close exists).

### B. Leakage Prevention (Strict Shift)
Because `target` at row $t$ is calculated using the close price at $t+1$, we must ensure that the model is only trained on features that were known *before* the price at $t+1$ was formed. 

To achieve this, the feature matrix $X$ is shifted forward by 1 bar:
```python
feature_cols = [c for c in df.columns if c not in ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "target", "TICKER", "INTERVAL"]]
X = df[feature_cols].shift(1)
```
By shifting the features by 1:
*   The features used at row $t$ to predict $C_{t+1} > C_t$ are $F_{t-1}$.
*   $F_{t-1}$ is calculated using prices up to and including $t-1$.
*   This creates a 1-bar buffer, ensuring no lookahead leakage of $C_t$ or $C_{t+1}$ prices into the training features.

---

## 3. Chronological Splitting

Financial time series exhibit autocorrelation and regime changes. Standard random train-test splitting causes data leakage. The framework enforces a **chronological split** (split-point at 80% mark) to preserve temporal ordering:

```python
split = int(len(df) * 0.8)
X_train = X.iloc[:split]
X_test  = X.iloc[split:]
y_train = y.iloc[:split]
y_test  = y.iloc[split:]
```
*   `shuffle=False` is set during preprocessing.
*   This splits the dataset into an 80% historical training block and a subsequent 20% validation block.

---

## 4. Hyperparameters & Evaluation

The XGBoost model uses the following parameters for hourly or daily classification:
```python
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
```

The performance is evaluated using the **ROC Area Under the Curve (ROC AUC)** score:
```python
probs_train = model.predict_proba(X_train)[:, 1]
probs_test  = model.predict_proba(X_test)[:, 1]
auc_train = roc_auc_score(y_train, probs_train)
auc_test  = roc_auc_score(y_test, probs_test)
```

---

## 5. Model Validation Tests

A dedicated test suite resides in [models/daily_model/xgboost/tests/](file:///d:/Files/Code/PocketQuant2/models/daily_model/xgboost/tests/) to prevent regression and ensure code validity:

*   `conftest.py`: Sets up a shared session-scoped `model_results` fixture that trains the model once on a small sample of AAPL data.
*   `test_leakage.py`:
    *   `test_no_future_leakage`: Verifies that the correlation of a feature with future prices ($C_{t+1}$) is not stronger than its correlation with current prices ($C_t$).
    *   `test_feature_shift`: Checks that the number of `NaN` rows in the feature columns is greater than or equal to the number of `NaN` rows in the target column (validating that the shift was executed).
*   `test_split.py`:
    *   `test_time_split_order`: Ensures that the maximum index of the training set is strictly less than the minimum index of the test set.
*   `test_auc.py`:
    *   `test_auc_not_random`: Verifies that the model learns a signal (checking that the maximum of AUC and 1-AUC is greater than 0.51).
    *   `test_probability_distribution`: Assures that model probabilities are not degenerate (std > 1e-3, and bounds are strictly within $[0, 1]$).
    *   `test_return_feature_direction`: Assures features like returns have a non-zero correlation with the target.
*   `test_labels.py`:
    *   `test_target_definition`: Randomly checks rows to verify that `target == 1` strictly corresponds to $C_{t+1} > C_t$.
