# src/core/use_cases/update_user.py
from src.core.entities.user import User


class UpdateUser:
    def __init__(self, user_repository):
        self.user_repository = user_repository

    def execute(self, user:User):
        self.user_repository.update_user(user)