from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, List, TypeVar, Generic, Type
from dataclasses import dataclass

T = TypeVar('T', bound='Asset')

@dataclass
class Asset(ABC):
    """
    Abstract base class representing a financial asset.
    
    Attributes:
        asset_id (str): Unique identifier for the asset
        name (str): Name of the asset
        symbol (str): Trading symbol
        current_price (float): Current market price
        purchase_price (float): Purchase price
        quantity (float): Number of units held
        purchase_date (datetime): When the asset was purchased
    """
    asset_id: str
    name: str
    symbol: str
    current_price: float
    purchase_price: float
    quantity: float
    purchase_date: datetime
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of the asset position."""
        return self.current_price * self.quantity
    
    @property
    def cost_basis(self) -> float:
        """Calculate the total cost basis of the asset position."""
        return self.purchase_price * self.quantity
    
    @property
    def profit_loss(self) -> float:
        """Calculate the current profit/loss of the asset position."""
        return self.market_value - self.cost_basis
    
    @property
    def profit_loss_pct(self) -> float:
        """Calculate the profit/loss percentage of the asset position."""
        return (self.profit_loss / self.cost_basis) * 100 if self.cost_basis != 0 else 0.0
    
    @abstractmethod
    def get_asset_type(self) -> str:
        """Return the type of the asset (to be implemented by subclasses)."""
        pass
    
    def to_dict(self) -> Dict:
        """Convert asset to dictionary representation."""
        return {
            'asset_id': self.asset_id,
            'name': self.name,
            'symbol': self.symbol,
            'current_price': self.current_price,
            'purchase_price': self.purchase_price,
            'quantity': self.quantity,
            'purchase_date': self.purchase_date.isoformat(),
            'asset_type': self.get_asset_type()
        }
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict) -> T:
        """Create an asset instance from a dictionary."""
        # This method will be overridden by subclasses
        pass
