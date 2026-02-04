# interfaces/controllers/product_controller.py

from src.core.use_cases.add_product import AddProduct
from src.core.use_cases.remove_product import RemoveProduct
from src.core.use_cases.update_inventory import UpdateInventory
from src.core.entities.product import Product

class ProductController:

    def __init__(self, add_product_use_case: AddProduct, remove_product_use_case: RemoveProduct, update_inventory_use_case: UpdateInventory):
        self.add_product_use_case = add_product_use_case
        self.remove_product_use_case = remove_product_use_case
        self.update_inventory_use_case = update_inventory_use_case

    def add_product(self, product_data: dict) -> None:
        product = Product(**product_data)
        self.add_product_use_case.execute(product)

    def remove_product(self, product_id: str) -> None:
        self.remove_product_use_case.execute(product_id)

    def update_inventory(self, product_id: str, quantity: int) -> None:
        self.update_inventory_use_case.execute(product_id, quantity)