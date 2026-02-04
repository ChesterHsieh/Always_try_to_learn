from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

@dataclass
class OrderItem:
    product_id: int
    quantity: int
    price: float

@dataclass
class Order:
    id: Optional[int]
    user_id: int
    items: List[OrderItem] = field(default_factory=list)
    total_amount: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"

    def calculate_total(self) -> None:
        self.total_amount = sum(item.price * item.quantity for item in self.items)

    def add_item(self, item: OrderItem) -> None:
        self.items.append(item)
        self.calculate_total()

    def remove_item(self, product_id: int) -> None:
        self.items = [item for item in self.items if item.product_id != product_id]
        self.calculate_total()