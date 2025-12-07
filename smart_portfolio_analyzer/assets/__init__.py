"""
Asset module containing all asset-related classes for the portfolio analyzer.

This module provides the base Asset class and its implementations for different
asset types like stocks, options, crypto, and forex.
"""

from .base_asset import Asset
from .stock_asset import StockAsset
from .option_asset import OptionAsset
from .crypto_asset import CryptoAsset
from .forex_asset import ForexAsset

# Re-export all public classes and functions
__all__ = [
    'Asset',
    'StockAsset',
    'OptionAsset',
    'CryptoAsset',
    'ForexAsset'
]
