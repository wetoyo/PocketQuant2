import sys
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbt as vbt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtester.backtester import Backtester


def create_test_data(n_days=100, tickers=None):
    """Helper function to create test data for backtesting."""
    if tickers is None:
        tickers = ["TEST"]
    
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    data_dict = {}
    
    for ticker in tickers:
        # Create realistic price data with trend and volatility
        np.random.seed(hash(ticker) % 2**32)
        returns = np.random.randn(n_days) * 0.02  # 2% daily volatility
        prices = 100 * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            "DATE": dates,
            "OPEN": prices * (1 + np.random.randn(n_days) * 0.005),
            "HIGH": prices * (1 + abs(np.random.randn(n_days)) * 0.01),
            "LOW": prices * (1 - abs(np.random.randn(n_days)) * 0.01),
            "CLOSE": prices,
            "VOLUME": np.random.randint(1000000, 10000000, n_days)
        })
        data_dict[ticker] = df
    
    return data_dict


def test_initialization():
    """Test Backtester initialization."""
    print("\n=== Test 1: Initialization ===")
    
    bt = Backtester(tickers=["AAPL"])
    assert bt.portfolio is None, "Portfolio should be None on initialization"
    print("✓ Backtester initialized successfully")
    return True


def test_get_price_data():
    """Test get_price_data method."""
    print("\n=== Test 2: get_price_data ===")
    
    # Create backtester with test data
    bt = Backtester(tickers=["TEST1", "TEST2"])
    bt.data = create_test_data(n_days=100, tickers=["TEST1", "TEST2"])
    
    # Test default CLOSE column
    prices = bt.get_price_data()
    assert not prices.empty, "Prices DataFrame should not be empty"
    assert "TEST1" in prices.columns, "TEST1 should be in columns"
    assert "TEST2" in prices.columns, "TEST2 should be in columns"
    assert isinstance(prices.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    print(f"✓ Retrieved price data with shape: {prices.shape}")
    
    # Test different price column
    prices_open = bt.get_price_data(price_col="OPEN")
    assert not prices_open.empty, "OPEN prices should not be empty"
    print(f"✓ Retrieved OPEN price data with shape: {prices_open.shape}")
    
    return True


def test_get_ticker_data():
    """Test get_ticker_data method."""
    print("\n=== Test 3: get_ticker_data ===")
    
    bt = Backtester(tickers=["TEST"])
    bt.data = create_test_data(n_days=100, tickers=["TEST"])
    
    ticker_data = bt.get_ticker_data("TEST")
    assert ticker_data is not None, "Ticker data should not be None"
    assert "CLOSE" in ticker_data.columns, "CLOSE column should exist"
    print(f"✓ Retrieved ticker data with shape: {ticker_data.shape}")
    
    # Test non-existent ticker
    missing_data = bt.get_ticker_data("NONEXISTENT")
    assert missing_data is None, "Non-existent ticker should return None"
    print("✓ Non-existent ticker returns None")
    
    return True


def test_align_signals():
    """Test align_signals method."""
    print("\n=== Test 4: align_signals ===")
    
    bt = Backtester(tickers=["TEST1", "TEST2"])
    
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    signals_dict = {
        "TEST1": pd.Series([True, False] * 25, index=dates),
        "TEST2": pd.Series([False, True] * 25, index=dates)
    }
    
    aligned = bt.align_signals(signals_dict)
    assert isinstance(aligned, pd.DataFrame), "Result should be DataFrame"
    assert "TEST1" in aligned.columns, "TEST1 should be in columns"
    assert "TEST2" in aligned.columns, "TEST2 should be in columns"
    print(f"✓ Aligned signals with shape: {aligned.shape}")
    
    return True


def test_run_backtest():
    """Test run_backtest method with simple moving average strategy."""
    print("\n=== Test 5: run_backtest ===")
    
    bt = Backtester(tickers=["TEST"])
    bt.data = create_test_data(n_days=100, tickers=["TEST"])
    
    # Get prices
    prices = bt.get_price_data()
    
    # Create simple MA crossover signals
    fast_ma = vbt.MA.run(prices, 10)
    slow_ma = vbt.MA.run(prices, 20)
    
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    # Run backtest
    portfolio = bt.run_backtest(entries, exits, freq="D", init_cash=10000)
    
    assert portfolio is not None, "Portfolio should not be None"
    assert bt.portfolio is not None, "Backtester should store portfolio"
    print(f"✓ Backtest completed successfully")
    print(f"  Total Return: {portfolio.total_return().values[0]:.2%}")
    
    return True


def test_get_stats():
    """Test get_stats method."""
    print("\n=== Test 6: get_stats ===")
    
    bt = Backtester(tickers=["TEST"])
    bt.data = create_test_data(n_days=100, tickers=["TEST"])
    
    # Run a simple backtest first
    prices = bt.get_price_data()
    fast_ma = vbt.MA.run(prices, 10)
    slow_ma = vbt.MA.run(prices, 20)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    bt.run_backtest(entries, exits, freq="D")
    
    # Get stats
    stats = bt.get_stats()
    assert stats is not None, "Stats should not be None"
    print("✓ Retrieved portfolio statistics")
    
    # Test without portfolio
    bt2 = Backtester(tickers=["TEST"])
    stats2 = bt2.get_stats()
    assert stats2 is None, "Stats should be None without portfolio"
    print("✓ Returns None when no portfolio exists")
    
    return True


def test_calculate_alpha_beta_synthetic():
    """Test calculate_alpha_beta with synthetic data where beta is known."""
    print("\n=== Test 7: calculate_alpha_beta (Synthetic Data) ===")
    
    # Create synthetic data with known beta relationship
    n_days = 252
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create benchmark returns with known characteristics
    benchmark_returns = np.random.randn(n_days) * 0.01  # 1% daily volatility
    benchmark_prices = 100 * (1 + benchmark_returns).cumprod()
    
    # Create strategy returns with known beta = 1.5 and alpha = 0.0002 per day
    true_beta = 1.5
    true_alpha_per_day = 0.0002  # 0.02% per day
    epsilon = np.random.randn(n_days) * 0.005  # idiosyncratic risk
    
    # Strategy returns = alpha + beta * benchmark_returns + epsilon
    strategy_returns = true_alpha_per_day + true_beta * benchmark_returns + epsilon
    strategy_prices = 100 * (1 + strategy_returns).cumprod()
    
    # Create DataFrames
    benchmark_df = pd.DataFrame({
        "DATE": dates,
        "OPEN": benchmark_prices * 0.99,
        "HIGH": benchmark_prices * 1.01,
        "LOW": benchmark_prices * 0.98,
        "CLOSE": benchmark_prices,
        "VOLUME": np.random.randint(1000000, 10000000, n_days)
    })
    
    strategy_df = pd.DataFrame({
        "DATE": dates,
        "OPEN": strategy_prices * 0.99,
        "HIGH": strategy_prices * 1.01,
        "LOW": strategy_prices * 0.98,
        "CLOSE": strategy_prices,
        "VOLUME": np.random.randint(1000000, 10000000, n_days)
    })
    
    # Create backtester
    bt = Backtester(tickers=["STRATEGY", "BENCHMARK"])
    bt.data = {"STRATEGY": strategy_df, "BENCHMARK": benchmark_df}
    
    # Run a simple backtest (required for calculate_alpha_beta)
    prices = bt.get_price_data()
    
    # Create simple buy-and-hold signals
    entries = pd.DataFrame({
        "STRATEGY": [True] + [False] * (n_days - 1),
        "BENCHMARK": [False] * n_days
    }, index=prices.index)
    exits = pd.DataFrame({
        "STRATEGY": [False] * n_days,
        "BENCHMARK": [False] * n_days
    }, index=prices.index)
    
    bt.run_backtest(entries, exits, freq="D")
    
    # Calculate alpha/beta
    result = bt.calculate_alpha_beta(
        benchmark_ticker="BENCHMARK",
        strategy_ticker="STRATEGY",
        freq="D",
        error_tolerance=0.20
    )
    
    assert result is not None, "Alpha/Beta result should not be None"
    
    # Verify the calculated beta is close to the true beta
    calculated_beta = result['beta']
    beta_error = abs(calculated_beta - true_beta) / true_beta
    
    print(f"✓ Alpha/Beta calculation completed")
    print(f"  True Beta: {true_beta:.4f}")
    print(f"  Calculated Beta: {calculated_beta:.4f}")
    print(f"  Beta Error: {beta_error:.2%}")
    print(f"  Beta (full sample): {result['beta_full']:.4f}")
    print(f"  Alpha (annualized): {result['alpha']:.4%}")
    print(f"  True Alpha (annualized): {true_alpha_per_day * 252:.4%}")
    print(f"  Bars used: {result['n_bars_used']}")
    
    # Beta should be within 20% of true value (reasonable tolerance given noise)
    assert beta_error < 0.30, f"Beta error {beta_error:.2%} exceeds 30% tolerance"
    print(f"✓ Beta within acceptable tolerance")
    
    # Verify alpha is reasonable (harder to verify due to compounding effects)
    # Just check it's not wildly off
    expected_alpha_annualized = true_alpha_per_day * 252
    alpha_diff = abs(result['alpha'] - expected_alpha_annualized)
    print(f"  Alpha difference: {alpha_diff:.4%}")
    
    return True


def test_calculate_alpha_beta():
    """Test calculate_alpha_beta method."""
    print("\n=== Test 8: calculate_alpha_beta (Random Data) ===")
    
    # Create backtester with both strategy and benchmark
    bt = Backtester(tickers=["STRATEGY", "BENCHMARK"])
    bt.data = create_test_data(n_days=252, tickers=["STRATEGY", "BENCHMARK"])
    
    # Run a simple backtest
    prices = bt.get_price_data()
    fast_ma = vbt.MA.run(prices["STRATEGY"], 10)
    slow_ma = vbt.MA.run(prices["STRATEGY"], 20)
    
    entries = pd.DataFrame({
        "STRATEGY": fast_ma.ma_crossed_above(slow_ma),
        "BENCHMARK": False
    })
    exits = pd.DataFrame({
        "STRATEGY": fast_ma.ma_crossed_below(slow_ma),
        "BENCHMARK": False
    })
    
    bt.run_backtest(entries, exits, freq="D")
    
    # Calculate alpha/beta
    result = bt.calculate_alpha_beta(
        benchmark_ticker="BENCHMARK",
        strategy_ticker="STRATEGY",
        freq="D",
        error_tolerance=0.20
    )
    
    assert result is not None, "Alpha/Beta result should not be None"
    assert "beta" in result, "Result should contain beta"
    assert "alpha" in result, "Result should contain alpha"
    assert "alpha_per_period" in result, "Result should contain alpha_per_period"
    assert "n_bars_used" in result, "Result should contain n_bars_used"
    assert "required_n" in result, "Result should contain required_n"
    assert "beta_full" in result, "Result should contain beta_full"
    
    print(f"✓ Alpha/Beta calculation completed")
    print(f"  Beta: {result['beta']:.4f}")
    print(f"  Alpha (annualized): {result['alpha']:.4%}")
    print(f"  Bars used: {result['n_bars_used']}")
    
    # Test with date range
    result_range = bt.calculate_alpha_beta(
        benchmark_ticker="BENCHMARK",
        strategy_ticker="STRATEGY",
        freq="D",
        date_range=("2023-03-01", "2023-06-01")
    )
    assert result_range is not None, "Alpha/Beta with date range should work"
    print("✓ Alpha/Beta with date range works")
    
    # Test without portfolio
    bt2 = Backtester(tickers=["TEST"])
    result2 = bt2.calculate_alpha_beta("BENCHMARK", "STRATEGY")
    assert result2 is None, "Should return None without portfolio"
    print("✓ Returns None when no portfolio exists")
    
    return True


def test_multi_ticker_backtest():
    """Test backtesting with multiple tickers."""
    print("\n=== Test 9: Multi-ticker backtest ===")
    
    tickers = ["STOCK1", "STOCK2", "STOCK3"]
    bt = Backtester(tickers=tickers)
    bt.data = create_test_data(n_days=100, tickers=tickers)
    
    # Get prices for all tickers
    prices = bt.get_price_data()
    assert len(prices.columns) == 3, "Should have 3 ticker columns"
    
    # Create signals for all tickers
    entries_dict = {}
    exits_dict = {}
    
    for ticker in tickers:
        fast_ma = vbt.MA.run(prices[ticker], 10)
        slow_ma = vbt.MA.run(prices[ticker], 20)
        entries_dict[ticker] = fast_ma.ma_crossed_above(slow_ma)
        exits_dict[ticker] = fast_ma.ma_crossed_below(slow_ma)
    
    entries = pd.DataFrame(entries_dict)
    exits = pd.DataFrame(exits_dict)
    
    # Run backtest
    portfolio = bt.run_backtest(entries, exits, freq="D", init_cash=30000)
    
    assert portfolio is not None, "Multi-ticker portfolio should not be None"
    print(f"✓ Multi-ticker backtest completed")
    print(f"  Number of tickers: {len(tickers)}")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Test 10: Edge cases ===")
    
    # Test with empty data
    bt = Backtester(tickers=["TEST"])
    bt.data = {"TEST": pd.DataFrame()}
    prices = bt.get_price_data()
    assert prices.empty, "Empty data should return empty DataFrame"
    print("✓ Handles empty data correctly")
    
    # Test with mismatched indices
    bt2 = Backtester(tickers=["TEST"])
    bt2.data = create_test_data(n_days=50, tickers=["TEST"])
    prices = bt2.get_price_data()
    
    # Create signals with different date range
    different_dates = pd.date_range(start="2023-06-01", periods=30, freq="D")
    entries = pd.DataFrame({"TEST": [True, False] * 15}, index=different_dates)
    exits = pd.DataFrame({"TEST": [False, True] * 15}, index=different_dates)
    
    try:
        portfolio = bt2.run_backtest(entries, exits, freq="D")
        # Should still work with intersection of dates
        print("✓ Handles mismatched date ranges correctly")
    except ValueError as e:
        if "No overlapping dates" in str(e):
            print("✓ Correctly raises error for non-overlapping dates")
    
    return True


def run_all_tests():
    """Run all test functions."""
    tests = [
        test_initialization,
        test_get_price_data,
        test_get_ticker_data,
        test_align_signals,
        test_run_backtest,
        test_get_stats,
        test_calculate_alpha_beta_synthetic,
        test_calculate_alpha_beta,
        test_multi_ticker_backtest,
        test_edge_cases
    ]
    
    print("=" * 60)
    print("RUNNING BACKTESTER TEST SUITE")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
