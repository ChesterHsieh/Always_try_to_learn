from typing import Optional
from ..entities.user import User
from src.interfaces import UserRepository

class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def register_user(self, user: User) -> User:
        return self.user_repository.create(user)

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        return self.user_repository.get_by_id(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        return self.user_repository.get_by_username(username)

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        user = self.user_repository.get_by_username(username)
        if user and user.check_password(password):
            return user
        return None

    def deactivate_user(self, user_id: int) -> bool:
        user = self.user_repository.get_by_id(user_id)
        if user:
            user.deactivate()
            self.user_repository.update(user)
            return True
        return False