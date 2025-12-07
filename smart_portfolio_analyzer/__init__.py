"""
SPARS (Smart Portfolio Analysis and Risk System)

A comprehensive tool for analyzing and managing investment portfolios.
"""

__version__ = "0.1.0"

from .portfolio import Portfolio
from .assets import Asset, StockAsset, CryptoAsset, ForexAsset, OptionAsset
from .risk_simulator import RiskSimulator
from .data_manager import DataManager
from .analytics import Analytics as PortfolioAnalyzer
from .market_data import market_data, update_portfolio_prices, load_sample_portfolio

__all__ = [
    'Portfolio',
    'Asset',
    'StockAsset',
    'CryptoAsset',
    'ForexAsset',
    'OptionAsset',
    'RiskSimulator',
    'DataManager',
    'PortfolioAnalyzer',
    'market_data',
    'update_portfolio_prices',
    'load_sample_portfolio'
]
__version__ = "0.1.0"
