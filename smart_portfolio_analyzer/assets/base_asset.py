import numpy as np
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, Optional

class Asset(ABC):
    """
    Abstract base class representing a financial asset.
    All specific asset types (stocks, bonds, etc.) should inherit from this class.
    """
    
    def __init__(self, 
                 asset_id: str, 
                 name: str, 
                 purchase_date: date, 
                 purchase_price: float, 
                 quantity: float):
        """
        Initialize an Asset with basic properties.
        
        Args:
            asset_id: Unique identifier for the asset
            name: Display name of the asset
            purchase_date: Date when the asset was purchased
            purchase_price: Purchase price per unit
            quantity: Number of units held
        """
        self._id = asset_id
        self._name = name
        self._purchase_date = purchase_date
        self._purchase_price = purchase_price
        self._quantity = quantity
        self._current_price: Optional[float] = None
        self._last_updated: Optional[date] = None

    @property
    @abstractmethod
    def asset_type(self) -> str:
        """Return the type of the asset (e.g., 'stock', 'bond')."""
        pass

    @property
    def id(self) -> str:
        """Get the asset's unique identifier."""
        return self._id

    @property
    def name(self) -> str:
        """Get the asset's display name."""
        return self._name

    @property
    def purchase_date(self) -> date:
        """Get the purchase date of the asset."""
        return self._purchase_date

    @property
    def purchase_price(self) -> float:
        """Get the purchase price per unit."""
        return self._purchase_price

    @property
    def quantity(self) -> float:
        """Get the number of units held."""
        return self._quantity

    @property
    def current_price(self) -> Optional[float]:
        """Get the current market price of the asset."""
        return self._current_price

    @current_price.setter
    def current_price(self, value: Optional[float]) -> None:
        """Set the current market price of the asset."""
        self._current_price = value

    @property
    def last_updated(self) -> Optional[date]:
        """Get when the asset was last updated."""
        return self._last_updated

    def update_price(self, new_price: float, as_of_date: date) -> None:
        """
        Update the current market price of the asset.
        
        Args:
            new_price: New market price
            as_of_date: Date of the price update
        """
        # TODO: Add validation for new_price > 0
        self._current_price = new_price
        self._last_updated = as_of_date

    def current_value(self) -> float:
        """
        Calculate the current market value of the asset.
        
        Returns:
            Current market value (price * quantity)
            
        Raises:
            ValueError: If current price is not set or invalid
        """
        if self._current_price is None:
            raise ValueError("Current price not set. Call update_price() first.")
            
        if not isinstance(self._current_price, (int, float)) or np.isnan(self._current_price):
            raise ValueError(f"Invalid current price: {self._current_price}")
            
        if not isinstance(self._quantity, (int, float)) or np.isnan(self._quantity):
            raise ValueError(f"Invalid quantity: {self._quantity}")
            
        value = self._current_price * self._quantity
        
        if value < 0:
            raise ValueError(f"Negative value calculated: {value} (price: {self._current_price}, quantity: {self._quantity})")
            
        return value

    def cost_basis(self) -> float:
        """
        Calculate the total cost basis of the asset.
        
        Returns:
            Total cost basis (purchase price * quantity)
        """
        return self._purchase_price * self._quantity

    def profit_loss(self) -> float:
        """
        Calculate the total profit/loss for this asset.
        
        Returns:
            Profit (positive) or loss (negative) amount
            
        Raises:
            ValueError: If current price is not set
        """
        return self.current_value() - self.cost_basis()

    def profit_loss_percent(self) -> float:
        """
        Calculate the profit/loss as a percentage of cost basis.
        
        Returns:
            Profit/loss as a decimal (e.g., 0.05 for 5% profit)
            
        Raises:
            ValueError: If current price is not set
            ZeroDivisionError: If cost basis is zero
        """
        cost = self.cost_basis()
        if cost == 0:
            raise ZeroDivisionError("Cannot calculate percentage with zero cost basis")
        return self.profit_loss() / cost

    def to_dict(self) -> Dict:
        """
        Convert asset properties to a dictionary for serialization.
        
        Returns:
            Dictionary containing asset properties
        """
        return {
            'id': self._id,
            'name': self._name,
            'purchase_date': self._purchase_date.isoformat(),
            'purchase_price': self._purchase_price,
            'quantity': self._quantity,
            'current_price': self._current_price,
            'last_updated': self._last_updated.isoformat() if self._last_updated else None,
            'asset_type': self.asset_type
        }
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> 'Asset':
        """
        Create an Asset instance from a dictionary.
        
        Args:
            data: Dictionary containing asset data
            
        Returns:
            New Asset instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        pass
