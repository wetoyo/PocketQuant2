# Framework for AI-Driven Backtesting: Rules & Best Practices

This document outlines the rigorous standards and systematic rules for designing, executing, and evaluating backtests for algorithmic trading strategies.

## 1. Data Integrity and Management
The quality of a backtest is bounded by the quality of the underlying data.

* **Survivorship Bias Avoidance:** Ensure the dataset includes assets that were delisted, went bankrupt, or were acquired. Using only currently active tickers leads to "look-ahead" success.
* **Adjusted Returns:** Account for dividends, stock splits, and corporate actions.
* **Timezone Consistency:** Align all data sources to a single timezone (UTC preferred) to avoid accidental "future-seeing" due to timestamp mismatches.
* **Data Resolution:** Match the data frequency (e.g., 1-minute, daily) to the strategy's execution window.

## 2. Preventing Data Leakage
Data leakage is the #1 cause of "paper millionaires" whose strategies fail in live markets.

* **The "Look-Ahead" Rule:** Ensure that at time $t$, the model only has access to information available at or before $t$.
* **Information Lag:** Incorporate a delay for fundamental data (e.g., earnings reports) to account for the time it takes for data to be disseminated and processed.
* **Train/Validation/Test Split:** * **In-Sample (IS):** Used for parameter optimization and training.
    * **Out-of-Sample (OOS):** Used for final validation. Never tune parameters based on OOS results.
* **Walk-Forward Analysis:** Periodically re-train the model on a sliding window to simulate how the AI would adapt to changing market regimes.

## 3. Realistic Transaction Modeling
Backtests often ignore the "friction" of the real world.

* **Slippage:** Model the difference between the expected price and the actual execution price, especially for large orders.
* **Commission & Fees:** Deduct exchange fees, broker commissions, and taxes from every trade.
* **Liquidity Constraints:** Do not assume you can execute a trade larger than a certain percentage (e.g., 10%) of the average daily volume (ADV) without moving the market.
* **Borrowing Costs:** For short strategies, account for the cost of borrowing shares (Hard-to-Borrow fees).

## 4. Performance Metrics & Risk Evaluation
Raw returns are a poor measure of success.

* **Sharpe & Sortino Ratios:** Evaluate risk-adjusted returns. Prefer Sortino for strategies with high positive volatility.
* **Maximum Drawdown (MDD):** Measure the largest peak-to-trough decline. Ensure it stays within the risk tolerance of the capital provider.
* **Calmar Ratio:** (Annualized Return / Max Drawdown). Useful for comparing long-term resilience.
* **Hit Rate vs. Win/Loss Ratio:** A high hit rate is meaningless if the average loss is significantly larger than the average win.
* **Profit Factor:** Total Gross Profit / Total Gross Loss. A value > 1.5 is typically sought.

## 5. Statistical Robustness
* **Overfitting Check:** If the strategy has more parameters than it has signal, it is likely overfit.
* **Monte Carlo Simulations:** Run the strategy against shuffled versions of the historical data to see if the "edge" was just a statistical fluke.
* **P-Value Testing:** Calculate the probability that the strategy's returns occurred by chance.
* **Sensitivity Analysis:** Vary the input parameters slightly. A robust strategy should not collapse if a moving average is changed from 20 to 21 days.

## 6. Documentation & Reproducibility
Every backtest must be documented to allow for peer review and auditing.

* **Version Control:** Use Git for all strategy code.
* **Log Everything:** Maintain a log of every trade, including entry/exit reasons, signal strength at the time, and the state of the model features.
* **Environment Specs:** Record the versions of libraries (Pandas, Scikit-learn, PyTorch) used to ensure identical results when re-run.

---
*Disclaimer: Past performance is not indicative of future results. Algorithmic trading involves substantial risk of loss.*