from typing import Dict, List, Optional, TypeVar, Tuple, Any, Type
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .assets import Asset, StockAsset, CryptoAsset, ForexAsset, OptionAsset

# Type aliases
FloatArray = npt.NDArray[np.float64]
ReturnArray = npt.NDArray[np.float64]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Portfolio')

@dataclass
class Portfolio:
    """
    A portfolio containing multiple financial assets with methods for portfolio analysis.
    
    Attributes:
        name (str): Name of the portfolio
        description (str): Optional description
        assets (List[Asset]): List of assets in the portfolio
        weights (Dict[str, float]): Weights of each asset by ticker
        risk_free_rate (float): Annual risk-free rate for calculations (default: 0.02)
        created_at (datetime): When the portfolio was created
    """
    name: str
    description: str = ""
    assets: List[Asset] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    risk_free_rate: float = 0.02
    created_at: datetime = field(default_factory=datetime.now)
    _covariance_matrix: Optional[np.ndarray] = field(init=False, default=None)
    _expected_returns: Optional[np.ndarray] = field(init=False, default=None)
    
    def __post_init__(self):
        """Initialize the portfolio and validate weights."""
        self._validate_weights()
    
    def _validate_weights(self) -> None:
        """Ensure weights sum to 1 (within floating point tolerance)."""
        if not self.weights:  # Skip validation if no weights are set yet
            return
            
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, rtol=1e-5):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def add_asset(self, asset: Asset, weight: Optional[float] = None) -> None:
        """
        Add an asset to the portfolio with an optional weight.
        
        Args:
            asset: The asset to add
            weight: Optional weight of the asset in the portfolio (0-1). 
                   If None, weight will be distributed equally among all assets.
            
        Time Complexity: O(n) where n is the number of assets (due to weight normalization)
        """
        if asset.ticker in {a.ticker for a in self.assets}:
            raise ValueError(f"Asset with ticker {asset.ticker} already exists in the portfolio")
            
        self.assets.append(asset)
        
        if weight is not None:
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight must be between 0 and 1, got {weight}")
            self.weights[asset.ticker] = weight
        else:
            # If no weight provided, distribute remaining weight equally
            n = len(self.assets)
            self.weights[asset.ticker] = 1.0 / n
            
        self._normalize_weights()
        self._invalidate_cache()
    
    def _normalize_weights(self) -> None:
        """Normalize weights to ensure they sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached calculations when portfolio changes."""
        self._covariance_matrix = None
        self._expected_returns = None
    
    def calculate_expected_return(self) -> float:
        """
        Calculate the expected return of the portfolio.
        
        Returns:
            float: The expected return as a decimal
            
        Time Complexity: O(n) where n is the number of assets
        """
        if not self.assets:
            return 0.0
            
        if self._expected_returns is None:
            self._expected_returns = np.array([asset.expected_return for asset in self.assets])
        
        weights = np.array([self.weights[asset.ticker] for asset in self.assets])
        return float(np.dot(weights, self._expected_returns))
    
    def calculate_volatility(self) -> float:
        """
        Calculate the annualized volatility of the portfolio.
        
        Returns:
            float: The annualized volatility as a decimal
            
        Time Complexity: O(n²) due to covariance matrix calculation
        """
        if not self.assets:
            return 0.0
            
        if self._covariance_matrix is None:
            self._calculate_covariance_matrix()
            
        weights = np.array([self.weights[asset.ticker] for asset in self.assets])
        portfolio_variance = weights.T @ self._covariance_matrix @ weights
        return float(np.sqrt(portfolio_variance))
    
    def _calculate_covariance_matrix(self) -> None:
        """Calculate and cache the covariance matrix of asset returns."""
        returns = np.column_stack([asset.historical_returns for asset in self.assets])
        self._covariance_matrix = np.cov(returns, rowvar=False) * 252  # Annualize
    
    def calculate_sharpe_ratio(self, risk_free_rate: Optional[float] = None) -> float:
        """
        Calculate the Sharpe ratio of the portfolio.
        
        Args:
            risk_free_rate: Optional override for risk-free rate
            
        Returns:
            float: The annualized Sharpe ratio
            
        Time Complexity: O(n²) due to volatility calculation
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        excess_return = self.calculate_expected_return() - risk_free_rate
        volatility = self.calculate_volatility()
        
        if volatility == 0:
            return 0.0
            
        return excess_return / volatility
    
    def rebalance(self, new_weights: Dict[str, float]) -> None:
        """
        Rebalance the portfolio to the specified weights.
        
        Args:
            new_weights: Dictionary mapping tickers to new weights
            
        Raises:
            ValueError: If weights don't match assets or don't sum to 1
        """
        if set(new_weights.keys()) != {asset.ticker for asset in self.assets}:
            raise ValueError("Weights must be provided for all assets")
            
        self.weights = new_weights
        self._validate_weights()
        self._invalidate_cache()
    
    def get_asset_weights(self) -> Dict[str, float]:
        """Return a copy of the current asset weights."""
        return self.weights.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the portfolio to a dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'assets': [asset.to_dict() for asset in self.assets],
            'weights': self.weights,
            'risk_free_rate': self.risk_free_rate,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """
        Create a Portfolio from a dictionary representation.
        
        Args:
            data: Dictionary containing portfolio data with keys:
                - name: str
                - description: str (optional)
                - assets: List[Dict] - List of asset dictionaries
                - weights: Dict[str, float] - Asset weights by ticker
                - risk_free_rate: float (optional, default=0.02)
                - created_at: str - ISO format datetime string
                
        Returns:
            Portfolio: A new Portfolio instance
            
        Time Complexity: O(n) where n is the number of assets
        """
        from .assets import StockAsset, CryptoAsset, ForexAsset, OptionAsset  # Import here to avoid circular imports
        
        # Create a new portfolio
        portfolio = cls(
            name=data['name'],
            description=data.get('description', ''),
            risk_free_rate=float(data.get('risk_free_rate', 0.02)),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.now()
        )
        
        # Add assets and weights
        asset_map = {}
        for asset_data in data.get('assets', []):
            asset_type = asset_data.get('asset_type')
            try:
                if asset_type == 'stock':
                    asset = StockAsset.from_dict(asset_data)
                elif asset_type == 'crypto':
                    asset = CryptoAsset.from_dict(asset_data)
                elif asset_type == 'forex':
                    asset = ForexAsset.from_dict(asset_data)
                elif asset_type == 'option':
                    asset = OptionAsset.from_dict(asset_data)
                else:
                    raise ValueError(f"Unknown asset type: {asset_type}")
                
                weight = data.get('weights', {}).get(asset.ticker, 0.0)
                portfolio.add_asset(asset, weight)
                
            except Exception as e:
                logger.error(f"Error loading asset {asset_data.get('ticker', 'unknown')}: {str(e)}")
                continue
                
        return portfolio
    
    def remove_asset(self, ticker: str) -> None:
        """
        Remove an asset from the portfolio by ticker.
        
        Args:
            ticker: The ticker symbol of the asset to remove
            
        Time Complexity: O(n) where n is the number of assets
        """
        # Find and remove the asset
        for i, asset in enumerate(self.assets):
            if asset.ticker == ticker:
                self.assets.pop(i)
                if ticker in self.weights:
                    del self.weights[ticker]
                self._normalize_weights()
                self._invalidate_cache()
                return
                
        raise ValueError(f"Asset with ticker {ticker} not found in portfolio")
    
    def update_asset(self, ticker: str, **updates) -> None:
        """
        Update an existing asset's properties.
        
        Args:
            ticker: The ticker symbol of the asset to update
            **updates: Key-value pairs of properties to update
            
        Time Complexity: O(n) where n is the number of assets
        """
        for asset in self.assets:
            if asset.ticker == ticker:
                for key, value in updates.items():
                    if hasattr(asset, key):
                        setattr(asset, key, value)
                    else:
                        raise AttributeError(f"{key} is not a valid attribute of {asset.__class__.__name__}")
                self._invalidate_cache()
                return
                
        raise ValueError(f"Asset with ticker {ticker} not found in portfolio")
    
    def get_asset(self, ticker: str) -> Asset:
        """
        Get an asset by its ticker symbol.
        
        Args:
            ticker: The ticker symbol of the asset to retrieve
            
        Returns:
            Asset: The requested asset
            
        Raises:
            ValueError: If no asset with the given ticker exists
            
        Time Complexity: O(n) where n is the number of assets
        """
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
        raise ValueError(f"Asset with ticker {ticker} not found in portfolio")
    
    def get_total_value(self) -> float:
        """
        Calculate the total market value of the portfolio.
        
        Returns:
            float: The total market value of all assets in the portfolio
            
        Time Complexity: O(n) where n is the number of assets
        """
        return sum(asset.current_value() for asset in self.assets if asset.current_value() is not None)
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """Get the allocation of each asset as a percentage of the total portfolio.
        
        Returns:
            Dict[str, float]: Dictionary mapping asset tickers to their allocation percentages
        """
        total_value = self.get_total_value()
        if total_value == 0:
            return {}
            
        asset_allocation = {}
        for asset in self.assets:
            # Only include assets with positive value
            asset_value = asset.current_value()  # Call as method
            if asset_value > 0:
                asset_allocation[asset.ticker] = (asset_value / total_value) * 100
                
        return asset_allocation
    
    def update_prices(self, data_manager=None) -> None:
        """
        Update the current prices of all assets in the portfolio.
        
        Args:
            data_manager: Optional DataManager instance to fetch current prices.
                         If not provided, will try to get it from session state if available.
        """
        if not self.assets:
            return
            
        # Get data manager from session state if not provided
        if data_manager is None:
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and 'data_manager' in st.session_state:
                    data_manager = st.session_state.data_manager
            except ImportError:
                # Streamlit is not available
                pass
                
        if data_manager is None:
            # If we still don't have a data manager, we can't update prices
            logger.warning("No data manager available to update prices")
            return
            
        # Get current prices for all assets
        tickers = [asset.ticker for asset in self.assets if hasattr(asset, 'ticker')]
        if not tickers:
            return
            
        try:
            # Get the most recent prices for all tickers
            # Using a 1-day lookback to ensure we get the latest price
            end_date = date.today()
            start_date = (end_date - timedelta(days=7)).strftime('%Y-%m-%d')  # 1 week lookback to ensure we get data
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Get historical data for the most recent period
            historical_data = data_manager.get_historical_prices(
                tickers, 
                start_date=start_date,
                end_date=end_date_str,
                interval='1d',
                source='yfinance'  # Use yfinance as it's more reliable for current data
            )
            
            # Update each asset with its most recent price
            for asset in self.assets:
                if hasattr(asset, 'ticker') and asset.ticker in historical_data:
                    df = historical_data[asset.ticker]
                    if not df.empty:
                        # Get the most recent price
                        latest_price = df['close'].iloc[-1]
                        asset.update_price(latest_price, end_date)
        except Exception as e:
            logger.error(f"Error updating asset prices: {str(e)}")
            raise
    
    def get_historical_returns(self, days: int = 30) -> Dict[str, List[float]]:
        """Simulate historical returns for the portfolio.
        
        In a real implementation, this would fetch actual historical data.
        
        Args:
            days: Number of days of historical data to generate
            
        Returns:
            Dict with 'dates' and 'returns' keys
        """
        
        # This is a simplified simulation
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.0005, 0.02, days)
        cumulative_returns = 100 * (1 + daily_returns).cumprod()
        
        return {
            'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                     for i in range(days, 0, -1)],
            'returns': cumulative_returns.tolist()
        }
    
    def to_dict(self) -> Dict:
        """Convert the portfolio to a dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'assets': {asset_id: asset.to_dict() for asset_id, asset in self.assets.items()},
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    def to_json(self, filepath: str = None) -> Optional[str]:
        """Serialize the portfolio to JSON."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            return None
        return json_str
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict) -> T:
        """Create a Portfolio instance from a dictionary."""
        portfolio = cls(
            name=data['name'],
            description=data.get('description', ''),
            created_at=datetime.fromisoformat(data['created_at'])
        )
        
        # Rebuild assets
        for asset_data in data.get('assets', {}).values():
            asset_type = asset_data.get('asset_type', '').upper()
            try:
                if asset_type == 'STOCK':
                    asset = StockAsset.from_dict(asset_data)
                elif asset_type == 'CRYPTO':
                    from .assets.crypto_asset import CryptoAsset
                    asset = CryptoAsset.from_dict(asset_data)
                elif asset_type == 'FOREX':
                    from .assets.forex_asset import ForexAsset
                    asset = ForexAsset.from_dict(asset_data)
                elif asset_type == 'OPTION':
                    from .assets.option_asset import OptionAsset
                    asset = OptionAsset.from_dict(asset_data)
                else:
                    logger.warning(f"Skipping unknown asset type: {asset_type}")
                    continue
                
                portfolio.assets[asset.asset_id] = asset
            except Exception as e:
                logger.error(f"Error creating {asset_type} asset from data: {asset_data}")
                logger.error(f"Error details: {str(e)}")
                continue
        
        portfolio.last_updated = datetime.fromisoformat(data.get('last_updated', data['created_at']))
        return portfolio
    
    @classmethod
    def from_json(cls: Type[T], json_str: str = None, filepath: str = None) -> T:
        """Create a Portfolio instance from a JSON string or file.
        
        Args:
            json_str: JSON string containing portfolio data
            filepath: Path to a JSON file containing portfolio data
            
        Returns:
            Portfolio: A new Portfolio instance
            
        Raises:
            ValueError: If neither json_str nor filepath is provided
            json.JSONDecodeError: If the JSON string is invalid
            FileNotFoundError: If the specified file doesn't exist
            
        Time Complexity: O(n) where n is the number of assets
        """
        if json_str is None and filepath is not None:
            with open(filepath, 'r') as f:
                json_str = f.read()
        
        if json_str is None:
            raise ValueError("Either json_str or filepath must be provided")
            
        data = json.loads(json_str)
        return cls.from_dict(data)
