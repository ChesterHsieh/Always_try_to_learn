# infrastructure/data_access/user_dao.py

from sqlalchemy.orm import Session
from src.interfaces.repositories.user_repository import UserRepository
from src.core.entities.user import User
from src.infrastructure.orm.models import UserModel


class UserDAO(UserRepository):

    def __init__(self, session: Session):
        self.session = session

    def add_user(self, user: User) -> None:
        user_model = UserModel(
            id=user.id,
            username=user.username,
            email=user.email,
            hashed_password=user.hashed_password
        )
        self.session.add(user_model)
        self.session.commit()

    def remove_user(self, user_id: str) -> None:
        user_model = self.session.query(UserModel).filter_by(id=user_id).first()
        if user_model:
            self.session.delete(user_model)
            self.session.commit()

    def update_user(self, user: User) -> None:
        user_model = self.session.query(UserModel).filter_by(id=user.id).first()
        if user_model:
            user_model.name = user.name
            user_model.email = user.email
            user_model.hashed_password = user.hashed_password
            self.session.commit()

    def get_user_by_id(self, user_id: str) -> User:
        user_model = self.session.query(UserModel).filter_by(id=user_id).first()
        if user_model:
            return User(
                id=user_model.id,
                name=user_model.name,
                email=user_model.email,
                hashed_password=user_model.hashed_password
            )
        return None

    def list_users(self) -> list:
        users = self.session.query(UserModel).all()
        return [
            User(
                id=user_model.id,
                username=user_model.username,
                email=user_model.email,
                hashed_password=user_model.hashed_password
            )
            for user_model in users
        ]