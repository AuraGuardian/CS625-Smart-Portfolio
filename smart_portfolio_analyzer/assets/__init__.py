"""
Asset module containing all asset-related classes for the portfolio analyzer.

This module provides the base Asset class and its implementations for different
asset types like stocks and bonds.
"""

from .base_asset import Asset
from .stock_asset import StockAsset
from .bond_asset import BondAsset

# Re-export all public classes and functions
__all__ = [
    'Asset',
    'StockAsset',
    'BondAsset'
]
