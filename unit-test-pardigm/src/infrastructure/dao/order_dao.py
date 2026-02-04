# infrastructure/data_access/order_dao.py

from sqlalchemy.orm import Session
from src.interfaces.repositories.order_repository import OrderRepository
from src.core.entities.order import Order
from src.infrastructure.orm.models import OrderModel


class OrderDAO(OrderRepository):

    def __init__(self, session: Session):
        self.session = session

    def add_order(self, order: Order) -> None:
        order_model = OrderModel(
            id=order.id,
            user_id=order.user_id,
            total_amount=order.total_amount,
            status=order.status
        )
        self.session.add(order_model)
        self.session.commit()

    def remove_order(self, order_id: str) -> None:
        order_model = self.session.query(OrderModel).filter_by(id=order_id).first()
        if order_model:
            self.session.delete(order_model)
            self.session.commit()

    def update_order(self, order: Order) -> None:
        order_model = self.session.query(OrderModel).filter_by(id=order.id).first()
        if order_model:
            order_model.user_id = order.user_id
            order_model.total_amount = order.total_amount
            order_model.status = order.status
            self.session.commit()

    def get_order_by_id(self, order_id: str) -> Order:
        order_model = self.session.query(OrderModel).filter_by(id=order_id).first()
        if order_model:
            return Order(
                id=order_model.id,
                user_id=order_model.user_id,
                total_amount=order_model.total_amount,
                status=order_model.status
            )
        return None

    def list_orders(self) -> list:
        orders = self.session.query(OrderModel).all()
        return [
            Order(
                id=order_model.id,
                user_id=order_model.user_id,
                total_amount=order_model.total_amount,
                status=order_model.status
            )
            for order_model in orders
        ]