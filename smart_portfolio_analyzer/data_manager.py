import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import requests
from urllib.parse import urljoin

from .portfolio import Portfolio
from .assets import StockAsset, BondAsset


class DataManager:
    """
    Handles data operations including reading from files, APIs, and other data sources.
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = 'data_cache'):
        """
        Initialize the DataManager.
        
        Args:
            api_key: API key for financial data services (e.g., Alpha Vantage, IEX Cloud)
            cache_dir: Directory to store cached data
        """
        self.api_key = api_key
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_historical_prices(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None,
        period: str = '1y',
        interval: str = '1d',
        source: str = 'yfinance'
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            period: Time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1d', '1wk', '1mo')
            source: Data source ('yfinance', 'alpha_vantage', 'iex')
            
        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        if not symbols:
            return {}
            
        if source == 'yfinance':
            return self._get_yfinance_data(symbols, start_date, end_date, period, interval)
        elif source == 'alpha_vantage' and self.api_key:
            return self._get_alpha_vantage_data(symbols, start_date, end_date, interval)
        elif source == 'iex' and self.api_key:
            return self._get_iex_data(symbols, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source or missing API key: {source}")
    
    def _get_yfinance_data(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None,
        period: str = '1y',
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data from Yahoo Finance."""
        data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                if start_date and end_date:
                    df = ticker.history(start=start_date, end=end_date, interval=interval)
                else:
                    df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    # Clean up the DataFrame
                    df = df.rename(columns={
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume',
                        'Dividends': 'dividends',
                        'Stock Splits': 'splits'
                    })
                    data[symbol] = df
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
        
        return data
    
    def _get_alpha_vantage_data(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data from Alpha Vantage."""
        # Implementation for Alpha Vantage API
        # Note: This is a placeholder - you would need to implement the actual API calls
        raise NotImplementedError("Alpha Vantage integration not implemented")
    
    def _get_iex_data(
        self, 
        symbols: List[str], 
        start_date: str = None, 
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data from IEX Cloud."""
        # Implementation for IEX Cloud API
        # Note: This is a placeholder - you would need to implement the actual API calls
        raise NotImplementedError("IEX Cloud integration not implemented")
    
    def get_dividend_history(
        self, 
        symbol: str, 
        start_date: str = None, 
        end_date: str = None
    ) -> pd.DataFrame:
        """Get dividend history for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            divs = ticker.dividends
            
            if start_date:
                divs = divs[divs.index >= start_date]
            if end_date:
                divs = divs[divs.index <= end_date]
                
            return pd.DataFrame(divs)
            
        except Exception as e:
            print(f"Error fetching dividend data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            company_info = {
                'symbol': symbol,
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
            return company_info
            
        except Exception as e:
            print(f"Error fetching company info for {symbol}: {str(e)}")
            return {}
    
    def get_bond_yield_curve(self, date: str = None) -> Dict[str, float]:
        """Get current or historical yield curve data."""
        # This is a simplified example - in practice, you would fetch this from a data provider
        # like Treasury.gov or FRED (Federal Reserve Economic Data)
        
        # Example yield curve (10-year Treasury yields as of a specific date)
        yield_curve = {
            '1m': 0.05,
            '3m': 0.08,
            '6m': 0.12,
            '1y': 0.18,
            '2y': 0.35,
            '5y': 0.85,
            '10y': 1.45,
            '30y': 2.10
        }
        
        return yield_curve
    
    def save_portfolio(self, portfolio: Portfolio, filename: str) -> None:
        """Save a portfolio to a JSON file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2)
    
    def load_portfolio(self, filename: str) -> Portfolio:
        """Load a portfolio from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return Portfolio.from_dict(data)
    
    def export_to_csv(self, data: Union[Dict, pd.DataFrame], filename: str) -> None:
        """
        Export data to a CSV file.
        
        Args:
            data: Data to export (DataFrame or dictionary of DataFrames)
            filename: Output filename
        """
        if isinstance(data, dict):
            # If it's a dictionary of DataFrames, create a directory and save each one
            os.makedirs(os.path.splitext(filename)[0], exist_ok=True)
            
            for key, df in data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    safe_key = "".join(c if c.isalnum() else "_" for c in str(key))
                    df.to_csv(os.path.join(os.path.splitext(filename)[0], f"{safe_key}.csv"))
        elif isinstance(data, pd.DataFrame):
            # If it's a single DataFrame, save it directly
            data.to_csv(filename)
    
    def get_benchmark_returns(
        self, 
        benchmark: str = '^GSPC',  # S&P 500 by default
        start_date: str = None, 
        end_date: str = None,
        period: str = '1y'
    ) -> pd.Series:
        """
        Get returns for a benchmark index.
        
        Args:
            benchmark: Benchmark symbol (e.g., '^GSPC' for S&P 500, '^IXIC' for NASDAQ)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            period: Time period if dates not specified
            
        Returns:
            Series of returns
        """
        try:
            if start_date and end_date:
                df = yf.download(benchmark, start=start_date, end=end_date, progress=False)
            else:
                df = yf.download(benchmark, period=period, progress=False)
            
            if not df.empty:
                return df['Adj Close'].pct_change().dropna()
            
        except Exception as e:
            print(f"Error fetching benchmark data for {benchmark}: {str(e)}")
        
        return pd.Series()
    
    def get_risk_free_rate(self) -> float:
        """
        Get the current risk-free rate (e.g., 10-year Treasury yield).
        
        Returns:
            Annualized risk-free rate as a decimal (e.g., 0.05 for 5%)
        """
        # In a real implementation, you would fetch this from a reliable source
        # This is a simplified example
        try:
            # Try to get the 10-year Treasury yield from Yahoo Finance
            ticker = yf.Ticker('^TNX')
            hist = ticker.history(period='1d')
            if not hist.empty:
                return hist['Close'].iloc[-1] / 100.0  # Convert from percentage to decimal
        except:
            pass
        
        # Default fallback value (e.g., 2.5%)
        return 0.025
