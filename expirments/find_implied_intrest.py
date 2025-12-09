import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from scraper.api_clients.YFinance import StockScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_implied_interest_rate(call_price, put_price, strike, S0, T):
    """
    Calculate implied risk-free interest rate using put-call parity.
    
    Put-Call Parity: C - P = S0 - K*exp(-r*T)
    Solving for r: r = -ln((S0 - C + P) / K) / T
    
    Parameters:
    -----------
    call_price : float
        Price of the European call option
    put_price : float
        Price of the European put option
    strike : float
        Strike price of options
    S0 : float
        Current stock price
    T : float
        Time to expiration (years)
    
    Returns:
    --------
    float : Implied risk-free rate (annual)
    """
    try:
        # Put-Call Parity: C - P = S0 - K*exp(-r*T)
        # Rearranging: K*exp(-r*T) = S0 - C + P
        # exp(-r*T) = (S0 - C + P) / K
        # -r*T = ln((S0 - C + P) / K)
        # r = -ln((S0 - C + P) / K) / T
        
        numerator = S0 - call_price + put_price
        if numerator <= 0 or strike <= 0:
            return np.nan
        
        ratio = numerator / strike
        if ratio <= 0:
            return np.nan
            
        implied_rate = -np.log(ratio) / T
        return implied_rate
        
    except Exception as e:
        logging.warning(f"Error calculating implied rate: {e}")
        return np.nan


def analyze_options_for_ticker(ticker, scraper):
    """
    Analyze options data for a single ticker and calculate implied interest rates.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    scraper : StockScraper
        Scraper instance with fetched options data
    
    Returns:
    --------
    pd.DataFrame : DataFrame with implied interest rates for each option pair
    """
    if ticker not in scraper.options_data:
        logging.warning(f"No options data found for {ticker}")
        return pd.DataFrame()
    
    # Get current stock price
    if ticker not in scraper.data or scraper.data[ticker].empty:
        logging.warning(f"No stock price data found for {ticker}")
        return pd.DataFrame()
    
    S0 = scraper.data[ticker]['CLOSE'].iloc[-1]
    logging.info(f"Current price for {ticker}: ${S0:.2f}")
    
    results = []
    today = pd.Timestamp.now()
    
    for expiration_date, chains in scraper.options_data[ticker].items():
        calls_df = chains['calls']
        puts_df = chains['puts']
        
        # Calculate time to expiration in years
        exp_datetime = pd.to_datetime(expiration_date)
        T = (exp_datetime - today).days / 365.0
        
        if T <= 0:
            continue
        
        # Merge calls and puts on strike price
        merged = pd.merge(
            calls_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest']],
            puts_df[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest']],
            on='strike',
            suffixes=('_call', '_put')
        )
        
        for _, row in merged.iterrows():
            strike = row['strike']
            
            # Use mid-price for better accuracy
            call_price = (row['bid_call'] + row['ask_call']) / 2 if row['bid_call'] > 0 and row['ask_call'] > 0 else row['lastPrice_call']
            put_price = (row['bid_put'] + row['ask_put']) / 2 if row['bid_put'] > 0 and row['ask_put'] > 0 else row['lastPrice_put']
            
            # Skip if prices are invalid
            if call_price <= 0 or put_price <= 0:
                continue
            
            # Calculate implied interest rate
            implied_rate = calculate_implied_interest_rate(call_price, put_price, strike, S0, T)
            
            if not np.isnan(implied_rate):
                # Conversion strategy: Buy stock + Buy put + Sell call
                # Cost basis = Stock price + Put premium - Call premium (received)
                conversion_cost = S0 + put_price - call_price
                
                # At expiration, conversion pays out the strike price
                conversion_payoff = strike
                
                # Profit from conversion
                conversion_profit = conversion_payoff - conversion_cost
                
                # Maximum interest rate you can pay on a loan and still be profitable
                # If you borrow conversion_cost at rate r for time T:
                # Interest paid = conversion_cost * r * T
                # Breakeven: conversion_profit = conversion_cost * r * T
                # r = conversion_profit / (conversion_cost * T)
                if conversion_cost > 0 and T > 0:
                    max_loan_rate = conversion_profit / (conversion_cost * T)
                    max_loan_rate_pct = max_loan_rate * 100
                else:
                    max_loan_rate = np.nan
                    max_loan_rate_pct = np.nan
                
                # Return on investment (annualized)
                if conversion_cost > 0:
                    roi_annualized = (conversion_profit / conversion_cost) / T
                    roi_annualized_pct = roi_annualized * 100
                else:
                    roi_annualized = np.nan
                    roi_annualized_pct = np.nan
                
                results.append({
                    'ticker': ticker,
                    'expiration': expiration_date,
                    'strike': strike,
                    'time_to_expiry_years': T,
                    'time_to_expiry_days': int(T * 365),
                    'stock_price': S0,
                    'call_price': call_price,
                    'put_price': put_price,
                    'call_volume': row['volume_call'],
                    'put_volume': row['volume_put'],
                    'call_open_interest': row['openInterest_call'],
                    'put_open_interest': row['openInterest_put'],
                    'implied_rate': implied_rate,
                    'implied_rate_pct': implied_rate * 100,
                    'conversion_cost': conversion_cost,
                    'conversion_payoff': conversion_payoff,
                    'conversion_profit': conversion_profit,
                    'max_loan_rate': max_loan_rate,
                    'max_loan_rate_pct': max_loan_rate_pct,
                    'roi_annualized': roi_annualized,
                    'roi_annualized_pct': roi_annualized_pct
                })
    
    return pd.DataFrame(results)


def main():
    """
    Main function to fetch options data and calculate implied interest rates.

    TODO: Put Call parity only holds for European options. Find better way to access European options data, Yfinance doesnt support it. 
    """
    # Configuration
    tickers = ["BNP.PA"]
    
    logging.info(f"Starting implied interest rate analysis for: {tickers}")
    
    # Setup scraper
    today = datetime.now().date()
    start_date = today.strftime("%Y-%m-%d")
    end_date = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    
    scraper = StockScraper(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        adjusted=True,
        fill_missing=True
    )
    
    # Fetch stock data
    logging.info("Fetching stock price data...")
    scraper.fetch_data()
    scraper.clean_data()
    
    # Fetch options data
    logging.info("Fetching options data...")
    scraper.fetch_options()
    
    # Analyze each ticker
    all_results = []
    for ticker in tickers:
        logging.info(f"\nAnalyzing options for {ticker}...")
        results_df = analyze_options_for_ticker(ticker, scraper)
        if not results_df.empty:
            all_results.append(results_df)
    
    if not all_results:
        logging.warning("No valid results found.")
        return
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Display summary statistics
    logging.info("\n" + "="*80)
    logging.info("SUMMARY STATISTICS")
    logging.info("="*80)
    
    for ticker in tickers:
        ticker_data = combined_df[combined_df['ticker'] == ticker]
        if ticker_data.empty:
            continue
            
        logging.info(f"\n{ticker}:")
        logging.info(f"  Number of option pairs analyzed: {len(ticker_data)}")
        logging.info(f"  Mean implied rate: {ticker_data['implied_rate_pct'].mean():.4f}%")
        logging.info(f"  Median implied rate: {ticker_data['implied_rate_pct'].median():.4f}%")
        logging.info(f"  Std dev: {ticker_data['implied_rate_pct'].std():.4f}%")
        logging.info(f"  Min: {ticker_data['implied_rate_pct'].min():.4f}%")
        logging.info(f"  Max: {ticker_data['implied_rate_pct'].max():.4f}%")
        
        # Conversion strategy metrics
        logging.info(f"\n  Conversion Strategy Metrics:")
        logging.info(f"  Average conversion cost: ${ticker_data['conversion_cost'].mean():.2f}")
        logging.info(f"  Average conversion profit: ${ticker_data['conversion_profit'].mean():.2f}")
        logging.info(f"  Mean max loan rate: {ticker_data['max_loan_rate_pct'].mean():.4f}%")
        logging.info(f"  Median max loan rate: {ticker_data['max_loan_rate_pct'].median():.4f}%")
        logging.info(f"  Mean annualized ROI: {ticker_data['roi_annualized_pct'].mean():.4f}%")
        
        # Show breakdown by expiration
        logging.info(f"\n  By Expiration:")
        exp_summary = ticker_data.groupby('expiration')['implied_rate_pct'].agg(['mean', 'median', 'count'])
        for exp, row in exp_summary.iterrows():
            logging.info(f"    {exp}: mean={row['mean']:.4f}%, median={row['median']:.4f}%, n={int(row['count'])}")
    
    # Save results to CSV
    output_file = current_dir / "implied_interest_rates.csv"
    combined_df.to_csv(output_file, index=False)
    logging.info(f"\n\nResults saved to: {output_file}")
    
    # Display top 10 best conversion opportunities (highest max loan rate)
    logging.info("\n" + "="*80)
    logging.info("TOP 10 BEST CONVERSION OPPORTUNITIES (by max loan rate)")
    logging.info("="*80)
    logging.info("These are conversions where you can borrow at the highest rates and still profit")
    top_conversions = combined_df.nlargest(10, 'max_loan_rate_pct')[[
        'ticker', 'expiration', 'strike', 'time_to_expiry_days', 
        'conversion_cost', 'conversion_profit', 'max_loan_rate_pct', 'roi_annualized_pct'
    ]]
    logging.info("\n" + top_conversions.to_string(index=False))
    
    # Display top 10 most liquid options (by volume)
    logging.info("\n" + "="*80)
    logging.info("TOP 10 MOST LIQUID OPTIONS (by total volume)")
    logging.info("="*80)
    combined_df['total_volume'] = combined_df['call_volume'] + combined_df['put_volume']
    top_liquid = combined_df.nlargest(10, 'total_volume')[[
        'ticker', 'expiration', 'strike', 'time_to_expiry_days', 
        'conversion_cost', 'max_loan_rate_pct', 'total_volume'
    ]]
    logging.info("\n" + top_liquid.to_string(index=False))


if __name__ == "__main__":
    main()
