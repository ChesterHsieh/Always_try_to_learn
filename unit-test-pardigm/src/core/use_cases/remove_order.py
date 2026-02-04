# src/core/use_cases/remove_order.py

class RemoveOrder:
    def __init__(self, order_repository):
        self.order_repository = order_repository

    def execute(self, order_id):
        self.order_repository.remove_order(order_id)