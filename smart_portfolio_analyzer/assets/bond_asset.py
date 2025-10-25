from datetime import date
from typing import Dict, Optional
from .base_asset import Asset


class BondAsset(Asset):
    """
    Represents a fixed-income bond asset in an investment portfolio.
    Inherits from the base Asset class and adds bond-specific functionality.
    """

    def __init__(
        self,
        asset_id: str,
        name: str,
        purchase_date: date,
        purchase_price: float,
        quantity: float,
        coupon_rate: float,
        maturity_date: date,
        face_value: float = 1000.0,
        payment_frequency: int = 2,
        issuer: str = "Corporate"
    ):
        """
        Initialize a new BondAsset instance.
        
        Args:
            asset_id: Unique identifier for the asset
            name: Display name of the bond
            purchase_date: Date when the bond was purchased
            purchase_price: Purchase price per bond (as a percentage of face value)
            quantity: Number of bonds held
            coupon_rate: Annual coupon rate as a decimal (e.g., 0.05 for 5%)
            maturity_date: Date when the bond matures
            face_value: Face value of the bond (default: $1000)
            payment_frequency: Number of coupon payments per year (default: 2 for semi-annual)
            issuer: Entity that issued the bond (e.g., 'US Treasury', 'Corporate')
        """
        super().__init__(asset_id, name, purchase_date, purchase_price, quantity)
        self._coupon_rate = coupon_rate
        self._maturity_date = maturity_date
        self._face_value = face_value
        self._payment_frequency = payment_frequency
        self._issuer = issuer
        self._credit_rating: Optional[str] = None
        self._yield_to_maturity: Optional[float] = None

    @property
    def asset_type(self) -> str:
        """Return the type of the asset."""
        return "bond"

    @property
    def coupon_rate(self) -> float:
        """Get the annual coupon rate as a decimal."""
        return self._coupon_rate

    @property
    def maturity_date(self) -> date:
        """Get the maturity date of the bond."""
        return self._maturity_date

    @property
    def face_value(self) -> float:
        """Get the face value of the bond."""
        return self._face_value

    @property
    def payment_frequency(self) -> int:
        """Get the number of coupon payments per year."""
        return self._payment_frequency

    @property
    def issuer(self) -> str:
        """Get the issuer of the bond."""
        return self._issuer

    @property
    def credit_rating(self) -> Optional[str]:
        """Get the credit rating of the bond."""
        return self._credit_rating

    @credit_rating.setter
    def credit_rating(self, value: Optional[str]) -> None:
        """Set the credit rating of the bond."""
        self._credit_rating = value

    @property
    def yield_to_maturity(self) -> Optional[float]:
        """Get the yield to maturity (YTM) of the bond."""
        return self._yield_to_maturity

    @yield_to_maturity.setter
    def yield_to_maturity(self, value: Optional[float]) -> None:
        """Set the yield to maturity (YTM) of the bond."""
        self._yield_to_maturity = value

    def years_to_maturity(self, as_of_date: Optional[date] = None) -> float:
        """
        Calculate the number of years until the bond matures.
        
        Args:
            as_of_date: Date to calculate from (defaults to today)
            
        Returns:
            Years to maturity as a float
        """
        as_of = as_of_date or date.today()
        delta = self._maturity_date - as_of
        return delta.days / 365.25

    def calculate_ytm(self, current_price: Optional[float] = None) -> float:
        """
        Calculate the yield to maturity (YTM) for the bond.
        
        Args:
            current_price: Current market price (if None, uses the asset's current_price)
            
        Returns:
            Yield to maturity as a decimal
            
        Note:
            This is a simplified calculation. For more accurate results, consider
            using a financial library or more sophisticated numerical methods.
        """
        price = current_price if current_price is not None else self._current_price
        if price is None:
            raise ValueError("Current price must be provided or set on the asset")
            
        # TODO: Implement more accurate YTM calculation
        # This is a simplified approximation
        years = self.years_to_maturity()
        if years <= 0:
            return 0.0
            
        coupon = self._face_value * self._coupon_rate
        return (coupon + (self._face_value - price) / years) / ((self._face_value + price) / 2)

    def current_yield(self) -> float:
        """
        Calculate the current yield of the bond.
        
        Returns:
            Current yield as a decimal
            
        Raises:
            ValueError: If current price is not set
        """
        if self._current_price is None:
            raise ValueError("Current price not set")
        return (self._face_value * self._coupon_rate) / self._current_price

    def duration(self, yield_change: float = 0.01) -> float:
        """
        Calculate the Macaulay duration of the bond.
        
        Args:
            yield_change: Small change in yield for approximation (default: 0.01 or 1%)
            
        Returns:
            Duration in years
            
        Note:
            This is a simplified calculation. For more accurate results, consider
            using a financial library.
        """
        # TODO: Implement more accurate duration calculation
        ytm = self.calculate_ytm()
        return (1 + ytm) / (self._payment_frequency * ytm) - (
            (1 + ytm) + (self.years_to_maturity() * (yield_change - ytm))) / (
            self._payment_frequency * ((1 + ytm) ** (self._payment_frequency * self.years_to_maturity()) - 1) + ytm * self._payment_frequency)

    def to_dict(self) -> Dict:
        """
        Convert the bond asset to a dictionary for serialization.
        
        Returns:
            Dictionary containing all bond properties
        """
        data = super().to_dict()
        data.update({
            'coupon_rate': self._coupon_rate,
            'maturity_date': self._maturity_date.isoformat(),
            'face_value': self._face_value,
            'payment_frequency': self._payment_frequency,
            'issuer': self._issuer,
            'credit_rating': self._credit_rating,
            'yield_to_maturity': self._yield_to_maturity
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'BondAsset':
        """
        Create a BondAsset instance from a dictionary.
        
        Args:
            data: Dictionary containing bond data
            
        Returns:
            New BondAsset instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            bond = cls(
                asset_id=data['id'],
                name=data['name'],
                purchase_date=date.fromisoformat(data['purchase_date']),
                purchase_price=data['purchase_price'],
                quantity=data['quantity'],
                coupon_rate=data['coupon_rate'],
                maturity_date=date.fromisoformat(data['maturity_date']),
                face_value=data.get('face_value', 1000.0),
                payment_frequency=data.get('payment_frequency', 2),
                issuer=data.get('issuer', 'Corporate')
            )
            
            # Set optional fields if they exist
            if 'current_price' in data:
                bond._current_price = data['current_price']
            if 'credit_rating' in data:
                bond.credit_rating = data['credit_rating']
            if 'yield_to_maturity' in data:
                bond.yield_to_maturity = data['yield_to_maturity']
            if 'last_updated' in data and data['last_updated']:
                bond._last_updated = date.fromisoformat(data['last_updated'])
                
            return bond
            
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid data: {e}")
