import pytest

from src.core.entities.order import Order
from src.core.use_cases.add_order import AddOrder
from unittest.mock import Mock, MagicMock

from src.infrastructure.dao.order_dao import OrderDAO


# Follow DIP principle, the test become super easy to write
# Mocking the repository, we could have
# 1) called test
# 2) mock object return value


@pytest.fixture
def order_dao():
    return MagicMock(spec=OrderDAO)

@pytest.fixture
def add_order_use_case(order_dao):
    return AddOrder(order_dao)

def adds_order_successfully(add_order_use_case, order_dao):
    order_data = {
        "id": "1",
        "user_id": "1",
        "total_amount": 1500.00,
        "status": "pending"
    }
    order = Order(**order_data)
    add_order_use_case.execute(order)
    order_dao.add_order.assert_called_once_with(order)

def raises_exception_for_invalid_order(add_order_use_case):
    order_data = {
        "id": "1",
        "user_id": "1",
        "total_amount": -1500.00,  # Invalid total amount
        "status": "pending"
    }
    order = Order(**order_data)
    with pytest.raises(ValueError):
        add_order_use_case.execute(order)