# src/core/use_cases/add_user.py
from src.core.entities.user import User


class AddUser:
    def __init__(self, user_repository):
        self.user_repository = user_repository

    def execute(self, user: User):
        self.user_repository.add_user(user)