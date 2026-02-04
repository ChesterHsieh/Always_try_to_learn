# infrastructure/data_access/product_dao.py

from sqlalchemy.orm import Session
from src.interfaces.repositories.product_repository import ProductRepository
from src.core.entities.product import Product
from src.infrastructure.orm.models import ProductModel


class ProductDAO(ProductRepository):

    def __init__(self, session: Session):
        self.session = session

    def add_product(self, product: Product) -> None:
        product_model = ProductModel(
            id=product.id,
            name=product.name,
            description=product.description,
            price=product.price,
            stock=product.stock
        )
        self.session.add(product_model)
        self.session.commit()

    def remove_product(self, product_id: str) -> None:
        product_model = self.session.query(ProductModel).filter_by(id=product_id).first()
        if product_model:
            self.session.delete(product_model)
            self.session.commit()

    def update_product(self, product: Product) -> None:
        product_model = self.session.query(ProductModel).filter_by(id=product.id).first()
        if product_model:
            product_model.name = product.name
            product_model.description = product.description
            product_model.price = product.price
            product_model.stock = product.stock
            self.session.commit()

    def get_product_by_id(self, product_id: str) -> Product:
        product_model = self.session.query(ProductModel).filter_by(id=product_id).first()
        if product_model:
            return Product(
                id=product_model.id,
                name=product_model.name,
                description=product_model.description,
                price=product_model.price,
                stock=product_model.stock
            )
        return None

    def list_products(self) -> list:
        products = self.session.query(ProductModel).all()
        return [
            Product(
                id=product_model.id,
                name=product_model.name,
                description=product_model.description,
                price=product_model.price,
                stock=product_model.stock
            )
            for product_model in products
        ]