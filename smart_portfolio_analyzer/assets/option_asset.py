from datetime import date, datetime
from typing import Dict, Optional, Literal
from .base_asset import Asset

class OptionAsset(Asset):
    """
    Represents an options contract in an investment portfolio.
    Inherits from the base Asset class and adds options-specific functionality.
    """

    def __init__(
        self,
        asset_id: str,
        name: str,
        purchase_date: date,
        purchase_price: float,  # Premium paid per contract
        quantity: float,  # Number of contracts
        underlying_ticker: str,
        option_type: Literal['call', 'put'],
        strike_price: float,
        expiration_date: date,
        multiplier: int = 100,  # Standard options contract multiplier
        exchange: str = "OPRA"  # Options Price Reporting Authority
    ):
        """
        Initialize a new OptionAsset instance.
        
        Args:
            asset_id: Unique identifier for the option
            name: Display name of the option (e.g., 'AAPL 150C 12/31/2023')
            purchase_date: Date when the option was purchased
            purchase_price: Premium paid per contract (in dollars)
            quantity: Number of contracts
            underlying_ticker: Ticker symbol of the underlying asset
            option_type: Type of option ('call' or 'put')
            strike_price: Strike price of the option
            expiration_date: Expiration date of the option
            multiplier: Contract multiplier (typically 100 for standard options)
            exchange: Exchange where the option is traded
        """
        super().__init__(asset_id, name, purchase_date, purchase_price, quantity)
        self._underlying_ticker = underlying_ticker
        self._option_type = option_type.lower()
        self._strike_price = strike_price
        self._expiration_date = expiration_date
        self._multiplier = multiplier
        self._exchange = exchange
        self._implied_volatility: Optional[float] = None
        self._delta: Optional[float] = None
        self._theta: Optional[float] = None
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptionAsset':
        """
        Create an OptionAsset instance from a dictionary.
        
        Args:
            data: Dictionary containing option asset data with keys:
                - asset_id: str
                - name: str
                - purchase_date: str (ISO format date)
                - purchase_price: float
                - quantity: float
                - underlying_ticker: str
                - option_type: str ('call' or 'put')
                - strike_price: float
                - expiration_date: str (ISO format date)
                - multiplier: int (optional, default=100)
                - exchange: str (optional, default="OPRA")
                
        Returns:
            OptionAsset: A new OptionAsset instance
        """
        # Convert string dates to date objects if they're strings
        def parse_date(date_str):
            if isinstance(date_str, str):
                try:
                    return datetime.strptime(date_str, '%Y-%m-%d').date()
                except ValueError:
                    return datetime.fromisoformat(date_str).date()
            return date_str
            
        purchase_date = parse_date(data['purchase_date'])
        expiration_date = parse_date(data['expiration_date'])
        
        return cls(
            asset_id=data['asset_id'],
            name=data['name'],
            purchase_date=purchase_date,
            purchase_price=float(data['purchase_price']),
            quantity=float(data['quantity']),
            underlying_ticker=data['underlying_ticker'],
            option_type=data['option_type'],
            strike_price=float(data['strike_price']),
            expiration_date=expiration_date,
            multiplier=int(data.get('multiplier', 100)),
            exchange=data.get('exchange', 'OPRA')
        )
        self._vega: Optional[float] = None
        self._gamma: Optional[float] = None
        self._open_interest: Optional[int] = None
        self._volume: Optional[int] = None

    @property
    def asset_type(self) -> str:
        """Return the type of the asset."""
        return "option"
        
    @property
    def ticker(self) -> str:
        """Get the underlying ticker symbol (for compatibility with code that expects a ticker attribute)."""
        return self._underlying_ticker

    @property
    def underlying_ticker(self) -> str:
        """Get the ticker symbol of the underlying asset."""
        return self._underlying_ticker

    @property
    def option_type(self) -> str:
        """Get the type of option ('call' or 'put')."""
        return self._option_type

    @property
    def strike_price(self) -> float:
        """Get the strike price of the option."""
        return self._strike_price

    @property
    def expiration_date(self) -> date:
        """Get the expiration date of the option."""
        return self._expiration_date

    @property
    def days_to_expiration(self) -> int:
        """Get the number of days until the option expires."""
        return (self._expiration_date - date.today()).days

    @property
    def intrinsic_value(self) -> float:
        """
        Calculate the intrinsic value of the option.
        Returns 0 if the option is out of the money.
        """
        if self._current_price is None:
            return 0.0
            
        if self._option_type == 'call':
            return max(0, self._current_price - self._strike_price)
        else:  # put
            return max(0, self._strike_price - self._current_price)

    @property
    def current_value(self) -> float:
        """
        Calculate the current market value of the option.
        
        Returns:
            Current market value (price * quantity * multiplier)
            
        Raises:
            ValueError: If current price is not set
        """
        if self._current_price is None:
            raise ValueError("Current price not set. Call update_price() first.")
        return self._current_price * self._quantity * self._multiplier

    def cost_basis(self) -> float:
        """
        Calculate the total cost basis of the option.
        
        Returns:
            Total cost basis (purchase price * quantity * multiplier)
        """
        return self._purchase_price * self._quantity * self._multiplier if self._multiplier != 0 else 0

    @property
    def time_value(self) -> float:
        """
        Calculate the time value of the option.
        Time value = Option Premium - Intrinsic Value
        """
        if self._current_price is None:
            return 0.0
            
        return max(0, self.current_value - (self.intrinsic_value * self._quantity * self._multiplier if self._multiplier != 0 else 0))

    @property
    def is_in_the_money(self) -> bool:
        """Check if the option is in the money."""
        if self._current_price is None:
            return False
            
        if self._option_type == 'call':
            return self._current_price > self._strike_price
        else:  # put
            return self._current_price < self._strike_price

    @property
    def is_at_the_money(self) -> bool:
        """Check if the option is at the money."""
        if self._current_price is None:
            return False
            
        return abs(self._current_price - self._strike_price) < 0.01  # Allow for small floating point differences

    @property
    def is_expired(self) -> bool:
        """Check if the option has expired."""
        return date.today() > self._expiration_date

    def update_greeks(self, delta: float, gamma: float, theta: float, vega: float) -> None:
        """
        Update the option's Greeks.
        
        Args:
            delta: Rate of change of the option price with respect to the price of the underlying
            gamma: Rate of change of delta with respect to the price of the underlying
            theta: Rate of change of the option price with respect to time
            vega: Rate of change of the option price with respect to volatility
        """
        self._delta = delta
        self._gamma = gamma
        self._theta = theta
        self._vega = vega

    def update_implied_volatility(self, iv: float) -> None:
        """
        Update the option's implied volatility.
        
        Args:
            iv: Implied volatility as a decimal (e.g., 0.20 for 20%)
        """
        self._implied_volatility = iv

    def update_market_data(self, price: float, volume: int, open_interest: int) -> None:
        """
        Update the option's market data.
        
        Args:
            price: Current market price of the option
            volume: Current trading volume
            open_interest: Current open interest
        """
        self.update_price(price, date.today())
        self._volume = volume
        self._open_interest = open_interest

    def to_dict(self) -> Dict:
        """Convert the option asset to a dictionary."""
        asset_dict = super().to_dict()
        asset_dict.update({
            'underlying_ticker': self._underlying_ticker,
            'option_type': self._option_type,
            'strike_price': self._strike_price,
            'expiration_date': self._expiration_date.isoformat(),
            'multiplier': self._multiplier,
            'exchange': self._exchange,
            'intrinsic_value': self.intrinsic_value,
            'time_value': self.time_value,
            'is_in_the_money': self.is_in_the_money,
            'is_at_the_money': self.is_at_the_money,
            'is_expired': self.is_expired,
            'implied_volatility': self._implied_volatility,
            'delta': self._delta,
            'gamma': self._gamma,
            'theta': self._theta,
            'vega': self._vega,
            'volume': self._volume,
            'open_interest': self._open_interest
        })
        return asset_dict
