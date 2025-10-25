from datetime import date
from typing import Dict, Optional
from .base_asset import Asset


class StockAsset(Asset):
    """
    Represents a stock asset in an investment portfolio.
    Inherits from the base Asset class and adds stock-specific functionality.
    """

    def __init__(self, 
                 asset_id: str, 
                 name: str, 
                 purchase_date: date,
                 purchase_price: float, 
                 quantity: float,
                 ticker: str,
                 exchange: str = "NYSE"):
        """
        Initialize a new StockAsset instance.
        
        Args:
            asset_id: Unique identifier for the asset
            name: Display name of the stock
            purchase_date: Date when the stock was purchased
            purchase_price: Purchase price per share
            quantity: Number of shares held
            ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
            exchange: Stock exchange where the stock is listed (default: 'NYSE')
        """
        super().__init__(asset_id, name, purchase_date, purchase_price, quantity)
        self._ticker = ticker
        self._exchange = exchange
        self._dividend_yield: Optional[float] = None
        self._sector: Optional[str] = None
        self._pe_ratio: Optional[float] = None
        self._beta: Optional[float] = 1.0

    @property
    def asset_type(self) -> str:
        """Return the type of the asset."""
        return "stock"

    @property
    def ticker(self) -> str:
        """Get the stock ticker symbol."""
        return self._ticker

    @property
    def exchange(self) -> str:
        """Get the stock exchange where the stock is listed."""
        return self._exchange

    @property
    def dividend_yield(self) -> Optional[float]:
        """Get the current dividend yield (as a decimal)."""
        return self._dividend_yield

    @dividend_yield.setter
    def dividend_yield(self, value: Optional[float]) -> None:
        """Set the dividend yield.
        
        Args:
            value: Dividend yield as a decimal (e.g., 0.05 for 5%)
        """
        if value is not None and value < 0:
            raise ValueError("Dividend yield cannot be negative")
        self._dividend_yield = value

    @property
    def sector(self) -> Optional[str]:
        """Get the sector the stock belongs to."""
        return self._sector

    @sector.setter
    def sector(self, value: Optional[str]) -> None:
        """Set the sector of the stock."""
        self._sector = value

    @property
    def pe_ratio(self) -> Optional[float]:
        """Get the price-to-earnings (P/E) ratio."""
        return self._pe_ratio

    @pe_ratio.setter
    def pe_ratio(self, value: Optional[float]) -> None:
        """Set the price-to-earnings (P/E) ratio."""
        if value is not None and value <= 0:
            raise ValueError("P/E ratio must be positive")
        self._pe_ratio = value

    def annual_dividend_income(self) -> float:
        """
        Calculate the annual dividend income from this stock.
        
        Returns:
            Annual dividend income in the portfolio's base currency
            
        Raises:
            ValueError: If dividend yield is not set
        """
        if self._dividend_yield is None:
            raise ValueError("Dividend yield not set")
        return self.current_value() * self._dividend_yield

    def to_dict(self) -> Dict:
        """
        Convert the stock asset to a dictionary for serialization.
        
        Returns:
            Dictionary containing all stock properties
        """
        data = super().to_dict()
        data.update({
            'ticker': self._ticker,
            'exchange': self._exchange,
            'dividend_yield': self._dividend_yield,
            'sector': self._sector,
            'pe_ratio': self._pe_ratio
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'StockAsset':
        """
        Create a StockAsset instance from a dictionary.
        
        Args:
            data: Dictionary containing stock data
            
        Returns:
            New StockAsset instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            stock = cls(
                asset_id=data['id'],
                name=data['name'],
                purchase_date=date.fromisoformat(data['purchase_date']),
                purchase_price=data['purchase_price'],
                quantity=data['quantity'],
                ticker=data['ticker'],
                exchange=data.get('exchange', 'NYSE')
            )
            
            # Set optional fields if they exist
            if 'current_price' in data:
                stock._current_price = data['current_price']
            if 'dividend_yield' in data:
                stock.dividend_yield = data['dividend_yield']
            if 'sector' in data:
                stock.sector = data['sector']
            if 'pe_ratio' in data:
                stock.pe_ratio = data['pe_ratio']
            if 'last_updated' in data and data['last_updated']:
                stock._last_updated = date.fromisoformat(data['last_updated'])
                
            return stock
            
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data: {e}")
