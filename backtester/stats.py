"""
Statistical robustness utilities for backtest validation.

Implements the checks required by instructions/rules.md §5:
  - p_value_test          Permutation test: is the edge statistically significant?
  - monte_carlo           Return-shuffling simulation: Sharpe / drawdown distribution
  - sensitivity_analysis  Parameter sweep: does performance collapse on small changes?

All three accept a vectorbt Portfolio, a pd.Series, or a raw np.ndarray of
per-period returns as the `source` argument (except sensitivity_analysis which
takes a callable and a param grid).
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_returns(source) -> np.ndarray:
    """
    Normalise source to a 1-D float ndarray of per-period returns (NaNs dropped).
    Accepts: vectorbt Portfolio | pd.Series | pd.DataFrame (first column) | np.ndarray.
    """
    try:
        # vectorbt Portfolio
        rets = source.returns()
        if isinstance(rets, pd.DataFrame):
            rets = rets.iloc[:, 0]
        return rets.dropna().to_numpy(dtype=float)
    except AttributeError:
        pass

    if isinstance(source, pd.DataFrame):
        source = source.iloc[:, 0]
    if isinstance(source, pd.Series):
        return source.dropna().to_numpy(dtype=float)
    if isinstance(source, np.ndarray):
        arr = source.astype(float).ravel()
        return arr[~np.isnan(arr)]

    raise TypeError(f"Unsupported source type: {type(source)}")


def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (risk-free rate = 0)."""
    if len(returns) < 2:
        return 0.0
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (negative number, e.g. -0.25 = -25%)."""
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
    return float(dd.min())


def _total_return(returns: np.ndarray) -> float:
    """Compounded total return."""
    return float(np.prod(1.0 + returns) - 1.0)


# ---------------------------------------------------------------------------
# p_value_test
# ---------------------------------------------------------------------------

def p_value_test(
    source,
    n_permutations: int = 10_000,
    metric: str = "sharpe",
    periods_per_year: int = 252,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    Permutation test: probability that the strategy's edge occurred by chance.

    Shuffles the return order `n_permutations` times and counts how often the
    shuffled metric meets or exceeds the actual metric.  A low p-value (< 0.05)
    means the edge is unlikely to be a random artefact of the path ordering.

    Args:
        source:           vectorbt Portfolio, pd.Series, pd.DataFrame, or np.ndarray
                          of per-period returns.
        n_permutations:   Random shuffles to run (default 10 000).
        metric:           "sharpe" or "total_return".
        periods_per_year: For Sharpe annualisation — 252 (daily), 52 (weekly), 12 (monthly).
        random_state:     Integer seed for reproducibility.

    Returns:
        dict:
            p_value       – fraction of shuffles that matched or beat the actual metric
            actual_metric – observed value of the chosen metric
            median_null   – median metric across all shuffled paths
            std_null      – std of metric across all shuffled paths

    Rule of thumb (rules.md §5):
        p_value < 0.05  →  statistically significant at 95% confidence
        p_value < 0.01  →  statistically significant at 99% confidence
    """
    if metric not in ("sharpe", "total_return"):
        raise ValueError(f"metric must be 'sharpe' or 'total_return', got '{metric}'")

    rng = np.random.default_rng(random_state)
    returns = _extract_returns(source)

    if metric == "sharpe":
        actual = _sharpe(returns, periods_per_year)
        compute = lambda r: _sharpe(r, periods_per_year)
    else:
        actual = _total_return(returns)
        compute = _total_return

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = compute(rng.permutation(returns))

    return {
        "p_value": float(np.mean(null >= actual)),
        "actual_metric": float(actual),
        "median_null": float(np.median(null)),
        "std_null": float(np.std(null)),
    }


# ---------------------------------------------------------------------------
# monte_carlo
# ---------------------------------------------------------------------------

def monte_carlo(
    source,
    n_simulations: int = 1_000,
    periods_per_year: int = 252,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Monte Carlo robustness test via return-order shuffling.

    Randomly reorders the actual return series `n_simulations` times to produce
    alternative equity paths and reports the distribution of Sharpe ratios, max
    drawdowns, and total returns across those paths.

    The actual strategy path is appended as the last row (simulation_id == -1).
    After running, print() output shows where the actual strategy falls in the
    simulated distribution — a genuine edge should sit in the upper percentiles.

    Args:
        source:           vectorbt Portfolio, pd.Series, pd.DataFrame, or np.ndarray.
        n_simulations:    Number of shuffled paths (default 1 000).
        periods_per_year: For Sharpe annualisation.
        random_state:     Seed for reproducibility.

    Returns:
        pd.DataFrame with columns:
            simulation_id   (int, -1 = actual path)
            sharpe          annualized Sharpe of the path
            max_drawdown    max peak-to-trough drawdown (negative float)
            total_return    compounded total return

    Interpretation (rules.md §5):
        sharpe_pct > 0.80  →  actual Sharpe beats 80% of random paths  (strong edge)
        sharpe_pct < 0.50  →  strategy offers no path-independent edge  (likely overfit)
    """
    rng = np.random.default_rng(random_state)
    returns = _extract_returns(source)

    rows = []
    for i in range(n_simulations):
        r = rng.permutation(returns)
        rows.append({
            "simulation_id": i,
            "sharpe": _sharpe(r, periods_per_year),
            "max_drawdown": _max_drawdown(r),
            "total_return": _total_return(r),
        })

    actual_sharpe = _sharpe(returns, periods_per_year)
    actual_mdd = _max_drawdown(returns)
    actual_tr = _total_return(returns)

    rows.append({
        "simulation_id": -1,
        "sharpe": actual_sharpe,
        "max_drawdown": actual_mdd,
        "total_return": actual_tr,
    })

    df = pd.DataFrame(rows)
    null = df[df["simulation_id"] != -1]

    sharpe_pct = float((null["sharpe"] <= actual_sharpe).mean())
    mdd_pct = float((null["max_drawdown"] >= actual_mdd).mean())  # higher MDD magnitude is worse

    print(
        f"[monte_carlo] Actual Sharpe {actual_sharpe:.3f} "
        f"beats {sharpe_pct:.1%} of {n_simulations} shuffled paths.\n"
        f"[monte_carlo] Actual MaxDD  {actual_mdd:.2%} "
        f"is shallower than {mdd_pct:.1%} of shuffled paths."
    )

    return df


# ---------------------------------------------------------------------------
# sensitivity_analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    run_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    param_grid: Dict[str, List[Any]],
) -> pd.DataFrame:
    """
    Parameter sensitivity sweep.

    Runs `run_fn` once per combination of values in `param_grid` and collects
    the returned performance metrics.  Use this to verify that the strategy
    doesn't collapse when a parameter moves by one tick (rules.md §5).

    Args:
        run_fn:     Callable(params: dict) -> dict.
                    Must return a dict with at least:
                        "sharpe"       (float)
                        "total_return" (float)
                        "max_drawdown" (float)
                    Any extra keys are preserved in the output DataFrame.
        param_grid: Dict mapping parameter name → list of values to sweep.
                    Example: {"fast_window": [8, 10, 12], "slow_window": [28, 30, 32]}

    Returns:
        pd.DataFrame where each row is one parameter combination and the metrics
        returned by run_fn. Failed runs have an "error" column with the exception
        message instead of metric columns.

    Example::

        from backtester.backtester import Backtester
        from backtester.stats import sensitivity_analysis
        import pandas as pd

        def run(params):
            bt = Backtester(
                tickers=["AAPL"],
                start_date="2023-01-01",
                end_date="2023-12-31",
                scraper_type="yfinance",
                features_options={
                    "ma_windows": [params["fast"], params["slow"]]
                }
            )
            bt.run_pipeline(save_folder_raw=None, save_folder_features=None)
            features = bt.get_ticker_features("AAPL")
            fast = features[f"MA_{params['fast']}"]
            slow = features[f"MA_{params['slow']}"]
            entries = pd.DataFrame({"AAPL": (fast > slow) & (fast.shift(1) <= slow.shift(1))})
            exits   = pd.DataFrame({"AAPL": (fast < slow) & (fast.shift(1) >= slow.shift(1))})
            pf = bt.run_backtest(entries, exits, freq="D")
            rets = pf.returns().iloc[:, 0].dropna().to_numpy()
            return {
                "sharpe":       _sharpe(rets),
                "total_return": _total_return(rets),
                "max_drawdown": _max_drawdown(rets),
            }

        results = sensitivity_analysis(run, {"fast": [8, 10, 12], "slow": [28, 30, 32]})
        print(results.sort_values("sharpe", ascending=False))
    """
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    rows = []
    total = len(combos)
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        try:
            metrics = run_fn(params)
            rows.append({**params, **metrics})
        except Exception as exc:
            rows.append({**params, "error": str(exc)})
        print(f"[sensitivity_analysis] {idx + 1}/{total} — params: {params}")

    return pd.DataFrame(rows)
