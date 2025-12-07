from datetime import date
from typing import Optional
from .base_asset import Asset

class ForexAsset(Asset):
    """
    Represents a forex (foreign exchange) asset in an investment portfolio.
    Inherits from the base Asset class and adds forex-specific functionality.
    """

    def __init__(
        self,
        asset_id: str,
        name: str,
        purchase_date: date,
        purchase_price: float,
        quantity: float,
        symbol: str,
        base_currency: str,
        quote_currency: str = 'USD',
        exchange: Optional[str] = 'Forex',
    ):
        """
        Initialize a new ForexAsset instance.
        
        Args:
            asset_id: Unique identifier for the asset
            name: Display name of the asset (e.g., 'Euro', 'Japanese Yen')
            purchase_date: Date when the asset was purchased
            purchase_price: Purchase price per unit in the quote currency
            quantity: Number of units held
            symbol: Ticker symbol (e.g., 'EUR/USD', 'JPY/USD')
            base_currency: The base currency (e.g., 'EUR' in 'EUR/USD')
            quote_currency: The quote currency (e.g., 'USD' in 'EUR/USD')
            exchange: Exchange where the forex pair is traded (default: 'Forex')
        """
        super().__init__(asset_id, name, purchase_date, purchase_price, quantity)
        self._symbol = symbol.upper()
        self._base_currency = base_currency.upper()
        self._quote_currency = quote_currency.upper()
        self._exchange = exchange
        self._current_price = None
        self._last_updated = None

    @property
    def asset_type(self) -> str:
        """Return the type of the asset."""
        return "forex"

    @property
    def symbol(self) -> str:
        """Get the ticker symbol (e.g., 'EUR/USD')."""
        return self._symbol

    @property
    def base_currency(self) -> str:
        """Get the base currency."""
        return self._base_currency

    @property
    def quote_currency(self) -> str:
        """Get the quote currency."""
        return self._quote_currency

    @property
    def exchange(self) -> Optional[str]:
        """Get the exchange where the forex pair is traded."""
        return self._exchange

    def to_dict(self) -> dict:
        """Convert the asset to a dictionary."""
        asset_dict = super().to_dict()
        asset_dict.update({
            "symbol": self._symbol,
            "base_currency": self._base_currency,
            "quote_currency": self._quote_currency,
            "exchange": self._exchange,
            "asset_type": "forex"
        })
        return asset_dict
