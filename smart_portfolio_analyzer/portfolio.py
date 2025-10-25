from typing import Dict, List, Optional, Union, TypeVar, Type
from datetime import datetime
import json
from dataclasses import dataclass, field
import numpy as np

from .assets import Asset, StockAsset, BondAsset

T = TypeVar('T', bound='Portfolio')

@dataclass
class Portfolio:
    """
    A portfolio containing multiple financial assets with methods for portfolio analysis.
    
    Attributes:
        name (str): Name of the portfolio
        description (str): Optional description
        assets (Dict[str, Asset]): Dictionary of assets keyed by asset_id
        created_at (datetime): When the portfolio was created
        last_updated (datetime): When the portfolio was last updated
    """
    name: str
    description: str = ""
    assets: Dict[str, Asset] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(init=False)
    
    def __post_init__(self):
        self.last_updated = self.created_at
    
    def add_asset(self, asset: Asset) -> None:
        """Add an asset to the portfolio."""
        if asset.asset_id in self.assets:
            raise ValueError(f"Asset with ID {asset.asset_id} already exists in the portfolio")
        self.assets[asset.asset_id] = asset
        self.last_updated = datetime.now()
    
    def remove_asset(self, asset_id: str) -> None:
        """Remove an asset from the portfolio."""
        if asset_id not in self.assets:
            raise KeyError(f"Asset with ID {asset_id} not found in the portfolio")
        del self.assets[asset_id]
        self.last_updated = datetime.now()
    
    def update_asset(self, asset_id: str, **updates) -> None:
        """Update an existing asset's properties."""
        if asset_id not in self.assets:
            raise KeyError(f"Asset with ID {asset_id} not found in the portfolio")
        
        asset = self.assets[asset_id]
        for key, value in updates.items():
            if hasattr(asset, key):
                setattr(asset, key, value)
        
        self.last_updated = datetime.now()
    
    def get_asset(self, asset_id: str) -> Asset:
        """Get an asset by its ID."""
        return self.assets.get(asset_id)
    
    def get_total_value(self) -> float:
        """Calculate the total market value of the portfolio."""
        return sum(asset.market_value for asset in self.assets.values())
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """Get the allocation of each asset as a percentage of the total portfolio."""
        total_value = self.get_total_value()
        if total_value == 0:
            return {}
            
        return {
            asset_id: (asset.market_value / total_value) * 100
            for asset_id, asset in self.assets.items()
        }
    
    def get_asset_type_allocation(self) -> Dict[str, float]:
        """Get the allocation by asset type (e.g., STOCK, BOND)."""
        total_value = self.get_total_value()
        if total_value == 0:
            return {}
            
        type_values = {}
        for asset in self.assets.values():
            asset_type = asset.get_asset_type()
            type_values[asset_type] = type_values.get(asset_type, 0) + asset.market_value
            
        return {t: (v / total_value) * 100 for t, v in type_values.items()}
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Get the allocation by sector (for stocks)."""
        total_value = self.get_total_value()
        if total_value == 0:
            return {}
            
        sector_values = {}
        for asset in self.assets.values():
            if isinstance(asset, StockAsset):
                sector = asset.sector or 'Unknown'
                sector_values[sector] = sector_values.get(sector, 0) + asset.market_value
            
        return {s: (v / total_value) * 100 for s, v in sector_values.items()}
    
    def get_historical_returns(self, days: int = 30) -> Dict[str, List[float]]:
        """
        Simulate historical returns for the portfolio.
        In a real implementation, this would fetch actual historical data.
        """
        # This is a simplified simulation
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.0005, 0.02, days)
        cumulative_returns = 100 * (1 + daily_returns).cumprod()
        
        return {
            'dates': [(datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d') 
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
            asset_type = asset_data.get('asset_type')
            if asset_type == 'STOCK':
                asset = StockAsset.from_dict(asset_data)
            elif asset_type == 'BOND':
                asset = BondAsset.from_dict(asset_data)
            else:
                continue  # Skip unknown asset types
                
            portfolio.assets[asset.asset_id] = asset
        
        portfolio.last_updated = datetime.fromisoformat(data.get('last_updated', data['created_at']))
        return portfolio
    
    @classmethod
    def from_json(cls: Type[T], json_str: str = None, filepath: str = None) -> T:
        """Create a Portfolio instance from a JSON string or file."""
        if json_str is None and filepath is not None:
            with open(filepath, 'r') as f:
                json_str = f.read()
        
        if json_str is None:
            raise ValueError("Either json_str or filepath must be provided")
            
        data = json.loads(json_str)
        return cls.from_dict(data)
