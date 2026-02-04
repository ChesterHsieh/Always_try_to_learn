# interfaces/repositories/order_repository.py

from abc import ABC, abstractmethod
from typing import List
from src.core.entities.order import Order

class OrderRepository(ABC):

    @abstractmethod
    def add_order(self, order: Order) -> None:
        pass

    @abstractmethod
    def remove_order(self, order_id: str) -> None:
        pass

    @abstractmethod
    def update_order(self, order: Order) -> None:
        pass

    @abstractmethod
    def get_order_by_id(self, order_id: str) -> Order:
        pass

    @abstractmethod
    def list_orders(self) -> List[Order]:
        pass