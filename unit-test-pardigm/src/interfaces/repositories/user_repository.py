# interfaces/repositories/user_repository.py

from abc import ABC, abstractmethod
from src.core.entities.user import User

class UserRepository(ABC):

    @abstractmethod
    def add_user(self, user: User) -> None:
        pass

    @abstractmethod
    def remove_user(self, user_id: str) -> None:
        pass

    @abstractmethod
    def update_user(self, user: User) -> None:
        pass

    @abstractmethod
    def get_user_by_id(self, user_id: str) -> User:
        pass

    @abstractmethod
    def list_users(self) -> list[User]:
        pass