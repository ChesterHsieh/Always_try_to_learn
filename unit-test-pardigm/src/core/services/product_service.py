from typing import List, Optional
from ..entities.product import Product
from src.interfaces import ProductRepository

class ProductService:
    def __init__(self, product_repository: ProductRepository):
        self.product_repository = product_repository

    def get_all_products(self) -> List[Product]:
        return self.product_repository.get_all()

    def get_product_by_id(self, product_id: int) -> Optional[Product]:
        return self.product_repository.get_by_id(product_id)

    def create_product(self, product: Product) -> Product:
        return self.product_repository.create(product)

    def update_product(self, product_id: int, product: Product) -> Optional[Product]:
        existing_product = self.product_repository.get_by_id(product_id)
        if existing_product:
            updated_product = Product(
                id=product_id,
                name=product.name,
                description=product.description,
                price=product.price,
                stock=product.stock
            )
            return self.product_repository.update(updated_product)
        return None

    def delete_product(self, product_id: int) -> bool:
        return self.product_repository.delete(product_id)