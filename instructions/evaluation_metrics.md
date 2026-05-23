# Metrics, Logging & Evaluation

This document outlines how backtest results are evaluated, how feature predictive power is quantified, and the statistical mechanics of the custom Alpha and Beta calculations.

---

## 1. Portfolio Statistics

Standard portfolio performance statistics are retrieved via the `get_stats()` method in `Backtester`. Under the hood, this calls `portfolio.stats()`. 

*   **Gotcha Prevention**: To avoid pandas aggregation warnings when dealing with multi-column (multi-ticker) portfolios, `get_stats()` explicitly sets `agg_func=None` by default. This forces vectorbt to return metrics for each ticker/column separately instead of trying to aggregate them.
*   **Common Metrics Returned**:
    *   `Total Return [%]`
    *   `Annualized Return [%]`
    *   `Sharpe Ratio`
    *   `Sortino Ratio`
    *   `Max Drawdown [%]`
    *   `Max Drawdown Duration`
    *   `Win Rate [%]`
    *   `Profit Factor`

---

## 2. Statistical Mechanics: Alpha and Beta

The `calculate_alpha_beta` method in `Backtester` calculates annualized Alpha and Beta against a benchmark ticker. It implements a dynamic sample-size lookback window based on standard error tolerance.

### The Algorithm and Mathematics
1.  **Returns Computation**: Price series are converted to percentage returns:
    $$R_s = \frac{P_{s, t} - P_{s, t-1}}{P_{s, t-1}}$$
    $$R_b = \frac{P_{b, t} - P_{b, t-1}}{P_{b, t-1}}$$
2.  **Initial OLS Regressions**: A preliminary regression on the entire dataset finds the full-sample beta ($\beta_{\text{full}}$), covariance ($\sigma_{s,b}$), and benchmark variance ($\sigma_m^2$):
    $$\beta_{\text{full}} = \frac{\text{Cov}(R_s, R_b)}{\text{Var}(R_b)}$$
    $$\alpha_{\text{full}} = \bar{R}_s - \beta_{\text{full}} \bar{R}_b$$
3.  **Standard Error & Residuals**: The standard deviation of the residuals ($\sigma_{\epsilon}$) is calculated:
    $$\epsilon_t = R_{s,t} - (\alpha_{\text{full}} + \beta_{\text{full}} R_{b,t})$$
    $$\sigma_{\epsilon} = \text{std}(\epsilon)$$
4.  **Required Sample Size ($N$)**: The target standard error ($SE$) is defined relative to the estimated Beta using `error_tolerance` (e.g., $20\%$ error tolerance):
    $$\text{Target } SE = \text{error\_tolerance} \times |\beta_{\text{full}}|$$
    The number of samples required to achieve this standard error is given by:
    $$N_{\text{required}} = \left( \frac{\sigma_{\epsilon}}{\sigma_m \times \text{Target } SE} \right)^2$$
    *   $N$ is bounded: $\min(N_{\text{required}}, \text{total\_bars})$ and capped at a minimum of $30$ bars.
5.  **Subset Regression**: The final $\beta$ and $\alpha_{\text{per\_period}}$ are calculated using only the last $N$ bars of the history.
6.  **Annualization**: Alpha is annualized based on the data frequency:
    $$\alpha_{\text{annualized}} = \alpha_{\text{per\_period}} \times \text{periods\_per\_year}$$
    Where `periods_per_year` is defined as:
    *   Daily (`"D"`): $252$
    *   Weekly (`"W"`): $52$
    *   Monthly (`"M"`): $12$

---

## 3. Feature Evaluation System

The `FeatureEvaluator` class in [research/evaluate_features.py](file:///d:/Files/Code/PocketQuant2/research/evaluate_features.py) assesses the predictive power of technical indicators before using them in strategies.

### A. Quantile Binning & Monotonicity
1.  **Forward Returns**: Computes forward returns over horizons $H$ (e.g., 5 or 15 bars):
    $$\text{fwd\_ret}_H = \frac{C_{t+H}}{C_t} - 1$$
2.  **Binning**: Features are split into quantiles (e.g., 5 bins) using a rank method to handle duplicate values gracefully:
    ```python
    self.df[f'{feat}_q'] = pd.qcut(self.df[feat].rank(method='first'), n_quantiles, labels=False)
    ```
3.  **Monotonicity Assessment**: Computes the mean forward return for each quantile. A monotonic relationship (e.g., higher quantiles consistently yield higher forward returns) indicates a stable directional signal.

### B. Signal Decay & Execution Delay
To model real-world slippage, the evaluator tests how signal strength decays if order entry is delayed by $d$ bars:
*   **Delayed Return**:
    $$\text{delayed\_fwd\_ret} = \text{fwd\_ret}_H \text{ shifted backward by } d \text{ bars}$$
*   **Spread Decay**: Measures the difference in mean return between the highest and lowest quantiles at delay $d=0$, $d=1$, and $d=2$. If the return spread drops by more than $50\%$ on a 1-bar delay, the feature is classified as **"Fragile"** instead of **"Realistic"**.

### C. Regime Analysis
Features are evaluated across different market regimes to find conditions where they excel:
*   **Time of Day Regime**: Splits intraday data into *Open* (before 11:00 AM), *Midday*, and *Close* (after 3:00 PM).
*   **Volatility Regime**: Splits days into *Low*, *Medium*, and *High* volatility groups based on the 20-bar rolling standard deviation of returns.
