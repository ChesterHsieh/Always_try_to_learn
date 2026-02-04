# src/core/use_cases/update_order.py
from src.core.entities.order import Order


class UpdateOrder:
    def __init__(self, order_repository):
        self.order_repository = order_repository

    def execute(self, order_data):
        order = Order(
            id=order_data["id"],
            user_id=order_data["user_id"],
            items=order_data["items"],
            total_amount=order_data["total_amount"],
            status=order_data["status"]
        )
        self.order_repository.update_order(order)