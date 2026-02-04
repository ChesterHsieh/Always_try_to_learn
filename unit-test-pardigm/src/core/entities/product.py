from dataclasses import dataclass
from typing import Optional

@dataclass
class Product:
    id: Optional[int]
    name: str
    description: str
    price: float
    stock: int

    def is_in_stock(self) -> bool:
        return self.stock > 0

    def update_stock(self, quantity: int) -> None:
        if quantity < 0 and abs(quantity) > self.stock:
            raise ValueError("Cannot reduce stock below zero")
        self.stock += quantity