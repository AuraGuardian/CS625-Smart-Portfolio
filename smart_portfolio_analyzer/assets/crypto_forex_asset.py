from datetime import date, datetime
from typing import Dict, Optional, Literal
from .base_asset import Asset

class CryptoForexAsset(Asset):
    """
    Represents a cryptocurrency or forex asset in an investment portfolio.
    Inherits from the base Asset class and adds crypto/forex-specific functionality.
    """

    def __init__(
        self,
        asset_id: str,
        name: str,
        purchase_date: date,
        purchase_price: float,
        quantity: float,
        symbol: str,
        asset_class: Literal['crypto', 'forex'],
        base_currency: str = 'USD',  # For forex pairs, the base currency (e.g., 'EUR' in 'EUR/USD')
        quote_currency: str = 'USD',  # For forex pairs, the quote currency (e.g., 'USD' in 'EUR/USD')
        exchange: Optional[str] = None,  # e.g., 'Coinbase', 'Binance', 'Forex'
        blockchain: Optional[str] = None,  # For crypto assets only
        contract_address: Optional[str] = None,  # For ERC-20 tokens and similar
        is_stablecoin: bool = False,  # For crypto assets only
        is_token: bool = False,  # For crypto assets only
    ):
        """
        Initialize a new CryptoForexAsset instance.
        
        Args:
            asset_id: Unique identifier for the asset
            name: Display name of the asset (e.g., 'Bitcoin', 'Euro')
            purchase_date: Date when the asset was purchased
            purchase_price: Purchase price per unit in the quote currency
            quantity: Number of units held
            symbol: Ticker symbol (e.g., 'BTC', 'EUR/USD')
            asset_class: Type of asset ('crypto' or 'forex')
            base_currency: For forex pairs, the base currency (e.g., 'EUR' in 'EUR/USD')
            quote_currency: The currency the price is quoted in (default: 'USD')
            exchange: Exchange where the asset is traded
            blockchain: For crypto assets, the underlying blockchain
            contract_address: For token assets, the smart contract address
            is_stablecoin: Whether the crypto asset is a stablecoin
            is_token: Whether the crypto asset is a token on another blockchain
        """
        super().__init__(asset_id, name, purchase_date, purchase_price, quantity)
        self._symbol = symbol.upper()
        self._asset_class = asset_class.lower()
        self._base_currency = base_currency.upper()
        self._quote_currency = quote_currency.upper()
        self._exchange = exchange
        self._blockchain = blockchain
        self._contract_address = contract_address
        self._is_stablecoin = is_stablecoin
        self._is_token = is_token
        self._market_cap: Optional[float] = None
        self._volume_24h: Optional[float] = None
        self._circulating_supply: Optional[float] = None
        self._all_time_high: Optional[float] = None
        self._all_time_low: Optional[float] = None
        self._price_change_24h: Optional[float] = None
        self._price_change_percentage_24h: Optional[float] = None

    @property
    def asset_type(self) -> str:
        """Return the type of the asset."""
        return f"{self._asset_class}_{'forex' if self._asset_class == 'forex' else 'crypto'}"

    @property
    def symbol(self) -> str:
        """Get the ticker symbol of the asset."""
        return self._symbol

    @property
    def asset_class(self) -> str:
        """Get the asset class ('crypto' or 'forex')."""
        return self._asset_class

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
        """Get the exchange where the asset is traded."""
        return self._exchange

    @property
    def is_stablecoin(self) -> bool:
        """Check if the crypto asset is a stablecoin."""
        return self._is_stablecoin

    @property
    def is_token(self) -> bool:
        """Check if the crypto asset is a token on another blockchain."""
        return self._is_token

    @property
    def blockchain(self) -> Optional[str]:
        """Get the underlying blockchain (for crypto assets)."""
        return self._blockchain

    @property
    def contract_address(self) -> Optional[str]:
        """Get the smart contract address (for token assets)."""
        return self._contract_address

    @property
    def market_cap(self) -> Optional[float]:
        """Get the market capitalization in USD."""
        return self._market_cap

    @property
    def volume_24h(self) -> Optional[float]:
        """Get the 24-hour trading volume in USD."""
        return self._volume_24h

    @property
    def circulating_supply(self) -> Optional[float]:
        """Get the circulating supply (for crypto assets)."""
        return self._circulating_supply

    def update_market_data(
        self,
        price: float,
        market_cap: Optional[float] = None,
        volume_24h: Optional[float] = None,
        circulating_supply: Optional[float] = None,
        price_change_24h: Optional[float] = None,
        price_change_percentage_24h: Optional[float] = None,
        all_time_high: Optional[float] = None,
        all_time_low: Optional[float] = None
    ) -> None:
        """
        Update the asset's market data.
        
        Args:
            price: Current price in the quote currency
            market_cap: Current market capitalization in USD
            volume_24h: 24-hour trading volume in USD
            circulating_supply: Current circulating supply (for crypto assets)
            price_change_24h: 24-hour price change in the quote currency
            price_change_percentage_24h: 24-hour price change percentage
            all_time_high: All-time high price in the quote currency
            all_time_low: All-time low price in the quote currency
        """
        self.update_price(price, date.today())
        self._market_cap = market_cap
        self._volume_24h = volume_24h
        self._circulating_supply = circulating_supply
        self._price_change_24h = price_change_24h
        self._price_change_percentage_24h = price_change_percentage_24h
        self._all_time_high = all_time_high if all_time_high is not None else self._all_time_high
        self._all_time_low = all_time_low if all_time_low is not None else self._all_time_low

    def to_dict(self) -> Dict:
        """Convert the crypto/forex asset to a dictionary."""
        asset_dict = super().to_dict()
        asset_dict.update({
            'symbol': self._symbol,
            'asset_class': self._asset_class,
            'base_currency': self._base_currency,
            'quote_currency': self._quote_currency,
            'exchange': self._exchange,
            'market_cap': self._market_cap,
            'volume_24h': self._volume_24h,
            'circulating_supply': self._circulating_supply,
            'price_change_24h': self._price_change_24h,
            'price_change_percentage_24h': self._price_change_percentage_24h,
            'all_time_high': self._all_time_high,
            'all_time_low': self._all_time_low,
            'is_stablecoin': self._is_stablecoin,
            'is_token': self._is_token,
            'blockchain': self._blockchain,
            'contract_address': self._contract_address
        })
        return asset_dict
