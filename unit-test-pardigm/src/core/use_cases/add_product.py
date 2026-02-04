# core/use_cases/add_product.py

from src.core.entities.product import Product
from src.interfaces.repositories.product_repository import ProductRepository

class AddProduct:

    def __init__(self, product_repository: ProductRepository):
        self.product_repository = product_repository

    def execute(self, product: Product) -> None:
        self.product_repository.add_product(product)