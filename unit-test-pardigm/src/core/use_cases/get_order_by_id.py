# src/core/use_cases/get_order_by_id.py

class GetOrderById:
    def __init__(self, order_repository):
        self.order_repository = order_repository

    def execute(self, order_id):
        return self.order_repository.get_order_by_id(order_id)