from datetime import datetime
from typing import Dict
from .base_asset import Asset

class StockAsset(Asset):
    """
    Represents a stock asset with additional stock-specific properties.
    
    Attributes:
        sector (str): The sector the stock belongs to (e.g., 'Technology', 'Healthcare')
        dividend_yield (float): Annual dividend yield as a percentage
        beta (float): Stock's beta (volatility relative to the market)
    """
    
    def __init__(
        self,
        asset_id: str,
        name: str,
        symbol: str,
        current_price: float,
        purchase_price: float,
        quantity: float,
        purchase_date: datetime,
        sector: str = "",
        dividend_yield: float = 0.0,
        beta: float = 1.0
    ):
        super().__init__(
            asset_id=asset_id,
            name=name,
            symbol=symbol,
            current_price=current_price,
            purchase_price=purchase_price,
            quantity=quantity,
            purchase_date=purchase_date
        )
        self.sector = sector
        self.dividend_yield = dividend_yield
        self.beta = beta
    
    def get_asset_type(self) -> str:
        """Return the type of the asset."""
        return "STOCK"
    
    def get_annual_dividend(self) -> float:
        """Calculate the annual dividend amount."""
        return (self.dividend_yield / 100) * self.current_price * self.quantity
    
    def to_dict(self) -> Dict:
        """Convert stock asset to dictionary representation."""
        data = super().to_dict()
        data.update({
            'sector': self.sector,
            'dividend_yield': self.dividend_yield,
            'beta': self.beta
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StockAsset':
        """Create a StockAsset instance from a dictionary."""
        return cls(
            asset_id=data['asset_id'],
            name=data['name'],
            symbol=data['symbol'],
            current_price=data['current_price'],
            purchase_price=data['purchase_price'],
            quantity=data['quantity'],
            purchase_date=datetime.fromisoformat(data['purchase_date']),
            sector=data.get('sector', ''),
            dividend_yield=data.get('dividend_yield', 0.0),
            beta=data.get('beta', 1.0)
        )
