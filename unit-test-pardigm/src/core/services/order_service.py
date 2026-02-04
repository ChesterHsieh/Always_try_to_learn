from typing import List, Optional
from core.entities.order import Order, OrderItem
from core.interfaces.repository import OrderRepository, ProductRepository


class OrderService:
    def __init__(self, order_repository: OrderRepository, product_repository: ProductRepository):
        self.order_repository = order_repository
        self.product_repository = product_repository

    def create_order(self, user_id: int, items: List[OrderItem]) -> Order:
        order = Order(id=None, user_id=user_id, items=items)
        order.calculate_total()
        created_order = self.order_repository.create(order)

        # Update stock for each product in the order
        for item in items:
            product = self.product_repository.get_by_id(item.product_id)
            if product:
                product.update_stock(-item.quantity)
                self.product_repository.update(product)

        return created_order

    def get_order_by_id(self, order_id: int) -> Optional[Order]:
        return self.order_repository.get_by_id(order_id)

    def get_orders_by_user_id(self, user_id: int) -> List[Order]:
        return self.order_repository.get_by_user_id(user_id)

    def cancel_order(self, order_id: int) -> bool:
        order = self.order_repository.get_by_id(order_id)
        if order and order.status == "pending":
            order.status = "canceled"
            self.order_repository.update(order)

            # Restore stock for each product in the canceled order
            for item in order.items:
                product = self.product_repository.get_by_id(item.product_id)
                if product:
                    product.update_stock(item.quantity)
                    self.product_repository.update(product)

            return True
        return False