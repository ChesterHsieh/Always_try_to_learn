# core/use_cases/add_order.py

from src.core.entities.order import Order
from src.interfaces.repositories.order_repository import OrderRepository

class AddOrder:

    def __init__(self, order_repository: OrderRepository):
        self.order_repository = order_repository

    def execute(self, order: Order) -> None:
        self.order_repository.add_order(order)