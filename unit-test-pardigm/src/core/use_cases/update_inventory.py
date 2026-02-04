# src/core/use_cases/update_inventory.py
from src.core.entities.product import Product


class UpdateInventory:
    def __init__(self, product_repository):
        self.product_repository = product_repository

    def execute(self, product_data):
        product = Product(
            id=product_data["id"],
            name=product_data["name"],
            description=product_data["description"],
            price=product_data["price"],
            stock=product_data["stock"]
        )
        self.product_repository.update_product(product)