"""
Market Data Demo for Smart Portfolio Analyzer

This script demonstrates how to use the market data functionality to:
1. Load a sample portfolio
2. Update portfolio prices with real-time data
3. Display portfolio performance metrics
"""
import json
from datetime import datetime, timedelta
import pandas as pd
from smart_portfolio_analyzer import (
    Portfolio, 
    market_data, 
    update_portfolio_prices, 
    load_sample_portfolio,
    PortfolioAnalyzer
)

def display_portfolio_summary(portfolio_data: dict):
    """Display a summary of the portfolio"""
    print("\n" + "="*50)
    print(f"Portfolio: {portfolio_data.get('name')}")
    print(f"Last Updated: {portfolio_data.get('last_updated', 'N/A')}")
    print("-"*50)
    
    # Display assets
    print("\nAssets:")
    print("-"*50)
    for asset in portfolio_data.get('assets', []):
        ticker = asset.get('ticker', 'N/A')
        asset_type = asset.get('asset_type', 'N/A').title()
        quantity = asset.get('quantity', 0)
        purchase_price = asset.get('purchase_price', 0)
        current_price = asset.get('current_price', 0)
        
        if purchase_price and current_price:
            pct_change = ((current_price - purchase_price) / purchase_price) * 100
            pct_str = f"{pct_change:+.2f}%"
        else:
            pct_str = "N/A"
            
        print(f"{ticker} ({asset_type}): {quantity} shares @ ${current_price:.2f} "
              f"[P/L: {pct_str}]")

def main():
    try:
        # Load the sample portfolio
        print("Loading sample portfolio...")
        portfolio_data = load_sample_portfolio()
        
        # Display initial portfolio
        display_portfolio_summary(portfolio_data)
        
        # Create a Portfolio instance
        portfolio = Portfolio.from_dict(portfolio_data)
        
        # Update prices with real-time data
        print("\nUpdating prices with real-time data...")
        updated_data = update_portfolio_prices(portfolio_data)
        
        # Display updated portfolio
        display_portfolio_summary(updated_data)
        
        # Get historical data for the first stock in the portfolio
        if updated_data.get('assets'):
            ticker = updated_data['assets'][0].get('ticker')
            if ticker:
                print(f"\nFetching historical data for {ticker}...")
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                
                hist_data = market_data.get_stock_aggregates(
                    ticker, 
                    start_date, 
                    end_date
                )
                
                if not hist_data.empty:
                    print(f"\nHistorical data for {ticker} (last 5 days):")
                    print(hist_data[['timestamp', 'open', 'high', 'low', 'close']].tail())
        
        # Get market status
        print("\nChecking market status...")
        market_status = market_data.get_market_status()
        print(f"Market is currently: {market_status.get('market', 'Unknown')}")
        
        # Calculate portfolio metrics
        analyzer = PortfolioAnalyzer(portfolio)
        print("\nPortfolio Metrics:")
        print("-"*50)
        print(f"Total Value: ${analyzer.calculate_total_value():,.2f}")
        print(f"Expected Return: {analyzer.calculate_expected_return() * 100:.2f}%")
        print(f"Volatility: {analyzer.calculate_volatility() * 100:.2f}%")
        print(f"Sharpe Ratio: {analyzer.calculate_sharpe_ratio():.2f}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
