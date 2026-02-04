# src/core/use_cases/remove_product.py

class RemoveProduct:
    def __init__(self, product_repository):
        self.product_repository = product_repository

    def execute(self, product_id):
        self.product_repository.remove_product(product_id)