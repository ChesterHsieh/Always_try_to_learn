# src/main.py
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.entities.order import Order
from src.infrastructure.orm.models import Base
from src.infrastructure.dao.product_dao import ProductDAO
from src.infrastructure.dao.user_dao import UserDAO
from src.infrastructure.dao.order_dao import OrderDAO

from src.core.use_cases.add_product import AddProduct
from src.core.use_cases.remove_product import RemoveProduct
from src.core.use_cases.update_inventory import UpdateInventory
from src.core.use_cases.add_user import AddUser
from src.core.use_cases.remove_user import RemoveUser
from src.core.use_cases.update_user import UpdateUser
from src.core.use_cases.add_order import AddOrder

from src.interfaces.controllers.product_controller import ProductController
from src.interfaces.controllers.user_controller import UserController

def remove_local_database():
    if os.path.exists('ecommerce.db'):
        os.remove('ecommerce.db')
def setup_database():
    engine = create_engine('sqlite:///ecommerce.db')
    Base.metadata.create_all(engine)
    return engine

def main():
    remove_local_database()
    engine = setup_database()
    Session = sessionmaker(bind=engine)
    session = Session()

    # Data Access Objects
    product_dao = ProductDAO(session)
    user_dao = UserDAO(session)
    order_dao = OrderDAO(session)

    # Use Cases
    add_product_use_case = AddProduct(product_dao)
    remove_product_use_case = RemoveProduct(product_dao)
    update_inventory_use_case = UpdateInventory(product_dao)

    add_user_use_case = AddUser(user_dao)
    remove_user_use_case = RemoveUser(user_dao)
    update_user_use_case = UpdateUser(user_dao)

    add_order_use_case = AddOrder(order_dao)

    # Controllers
    product_controller = ProductController(
        add_product_use_case, remove_product_use_case, update_inventory_use_case
    )
    user_controller = UserController(
        add_user_use_case, remove_user_use_case, update_user_use_case
    )

    # Sample data to demonstrate usage
    product_data = {
        "id": "1",
        "name": "Laptop",
        "description": "A high-end gaming laptop",
        "price": 1500.00,
        "stock": 10
    }
    product_controller.add_product(product_data)

    user_data = {
        "id": "1",
        "username": "Alice",
        "email": "alice@example.com",
        "hashed_password": "hashedpassword"
    }
    user_controller.add_user(user_data)

    order_data = {
        "id": "1",
        "user_id": "1",
        "total_amount": 1500.00,
        "status": "pending"
    }

    add_order_use_case.execute(Order(**order_data))

    print("Products:", product_dao.list_products())
    print("Users:", user_dao.list_users())
    print("Orders:", order_dao.list_orders())

if __name__ == "__main__":
    main()