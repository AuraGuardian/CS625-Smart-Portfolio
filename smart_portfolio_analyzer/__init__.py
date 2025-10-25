"""
Smart Portfolio Analyzer & Risk Simulator
A comprehensive tool for portfolio analysis and risk simulation.
"""

from .portfolio import Portfolio
from .risk_simulator import RiskSimulator
from .data_manager import DataManager
from .analytics import Analytics, ReturnType, PerformanceMetrics
from .assets import Asset, StockAsset, BondAsset

__all__ = [
    'Portfolio',
    'RiskSimulator',
    'DataManager',
    'Analytics',
    'ReturnType',
    'PerformanceMetrics',
    'Asset',
    'StockAsset',
    'BondAsset'
]

__version__ = "0.1.0"
