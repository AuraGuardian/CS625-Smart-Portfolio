from datetime import date
from typing import Optional
from .base_asset import Asset

class CryptoAsset(Asset):
    """
    Represents a cryptocurrency asset in an investment portfolio.
    Inherits from the base Asset class and adds crypto-specific functionality.
    """

    def __init__(
        self,
        asset_id: str,
        name: str,
        purchase_date: date,
        purchase_price: float,
        quantity: float,
        symbol: str,
        exchange: Optional[str] = None,
        blockchain: Optional[str] = None,
        contract_address: Optional[str] = None,
        is_stablecoin: bool = False,
        is_token: bool = False,
    ):
        """
        Initialize a new CryptoAsset instance.
        
        Args:
            asset_id: Unique identifier for the asset
            name: Display name of the asset (e.g., 'Bitcoin')
            purchase_date: Date when the asset was purchased
            purchase_price: Purchase price per unit in USD
            quantity: Number of units held
            symbol: Ticker symbol (e.g., 'BTC')
            exchange: Exchange where the asset is traded (e.g., 'Coinbase', 'Binance')
            blockchain: The underlying blockchain (e.g., 'Ethereum', 'Bitcoin')
            contract_address: For token assets, the smart contract address
            is_stablecoin: Whether the crypto asset is a stablecoin
            is_token: Whether the crypto asset is a token on another blockchain
        """
        super().__init__(asset_id, name, purchase_date, purchase_price, quantity)
        self._symbol = symbol.upper()
        self._exchange = exchange
        self._blockchain = blockchain
        self._contract_address = contract_address
        self._is_stablecoin = is_stablecoin
        self._is_token = is_token
        self._current_price = None
        self._last_updated = None

    @property
    def asset_type(self) -> str:
        """Return the type of the asset."""
        return "crypto"

    @property
    def symbol(self) -> str:
        """Get the ticker symbol."""
        return self._symbol

    @property
    def exchange(self) -> Optional[str]:
        """Get the exchange where the asset is traded."""
        return self._exchange

    @property
    def blockchain(self) -> Optional[str]:
        """Get the underlying blockchain."""
        return self._blockchain

    @property
    def contract_address(self) -> Optional[str]:
        """Get the smart contract address (for tokens)."""
        return self._contract_address

    @property
    def is_stablecoin(self) -> bool:
        """Check if the asset is a stablecoin."""
        return self._is_stablecoin

    @property
    def is_token(self) -> bool:
        """Check if the asset is a token on another blockchain."""
        return self._is_token

    def to_dict(self) -> dict:
        """Convert the asset to a dictionary."""
        asset_dict = super().to_dict()
        asset_dict.update({
            "symbol": self._symbol,
            "exchange": self._exchange,
            "blockchain": self._blockchain,
            "contract_address": self._contract_address,
            "is_stablecoin": self._is_stablecoin,
            "is_token": self._is_token,
            "asset_type": "crypto"
        })
        return asset_dict
