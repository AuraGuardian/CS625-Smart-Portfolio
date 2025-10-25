from datetime import datetime
from typing import Dict
from .base_asset import Asset

class BondAsset(Asset):
    """
    Represents a bond asset with bond-specific properties.
    
    Attributes:
        coupon_rate (float): Annual coupon rate as a percentage
        face_value (float): Face value of the bond
        maturity_date (datetime): When the bond matures
        credit_rating (str): Credit rating (e.g., 'AAA', 'BB+')
        payment_frequency (int): Number of coupon payments per year
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
        coupon_rate: float,
        face_value: float,
        maturity_date: datetime,
        credit_rating: str = "BBB",
        payment_frequency: int = 2
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
        self.coupon_rate = coupon_rate
        self.face_value = face_value
        self.maturity_date = maturity_date
        self.credit_rating = credit_rating
        self.payment_frequency = payment_frequency
    
    def get_asset_type(self) -> str:
        """Return the type of the asset."""
        return "BOND"
    
    def get_annual_coupon_payment(self) -> float:
        """Calculate the total annual coupon payment."""
        return (self.coupon_rate / 100) * self.face_value * self.quantity
    
    def get_current_yield(self) -> float:
        """Calculate the current yield based on current price."""
        if self.current_price == 0:
            return 0.0
        return (self.get_annual_coupon_payment() / (self.current_price * self.quantity)) * 100
    
    def get_years_to_maturity(self, as_of_date: datetime = None) -> float:
        """Calculate years remaining until maturity."""
        as_of = as_of_date or datetime.now()
        if self.maturity_date <= as_of:
            return 0.0
        return (self.maturity_date - as_of).days / 365.25
    
    def to_dict(self) -> Dict:
        """Convert bond asset to dictionary representation."""
        data = super().to_dict()
        data.update({
            'coupon_rate': self.coupon_rate,
            'face_value': self.face_value,
            'maturity_date': self.maturity_date.isoformat(),
            'credit_rating': self.credit_rating,
            'payment_frequency': self.payment_frequency
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BondAsset':
        """Create a BondAsset instance from a dictionary."""
        return cls(
            asset_id=data['asset_id'],
            name=data['name'],
            symbol=data['symbol'],
            current_price=data['current_price'],
            purchase_price=data['purchase_price'],
            quantity=data['quantity'],
            purchase_date=datetime.fromisoformat(data['purchase_date']),
            coupon_rate=data['coupon_rate'],
            face_value=data['face_value'],
            maturity_date=datetime.fromisoformat(data['maturity_date']),
            credit_rating=data.get('credit_rating', 'BBB'),
            payment_frequency=data.get('payment_frequency', 2)
        )
