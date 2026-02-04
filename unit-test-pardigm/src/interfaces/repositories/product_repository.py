# interfaces/repositories/product_repository.py

from abc import ABC, abstractmethod
from typing import List
from src.core.entities.product import Product

class ProductRepository(ABC):

    @abstractmethod
    def add_product(self, product: Product) -> None:
        pass

    @abstractmethod
    def remove_product(self, product_id: str) -> None:
        pass

    @abstractmethod
    def update_product(self, product: Product) -> None:
        pass

    @abstractmethod
    def get_product_by_id(self, product_id: str) -> Product:
        pass

    @abstractmethod
    def list_products(self) -> List[Product]:
        pass