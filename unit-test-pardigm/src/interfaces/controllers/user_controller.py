# interfaces/controllers/user_controller.py

from src.core.use_cases.add_user import AddUser
from src.core.use_cases.remove_user import RemoveUser
from src.core.use_cases.update_user import UpdateUser
from src.core.entities.user import User

class UserController:

    def __init__(self, add_user_use_case: AddUser, remove_user_use_case: RemoveUser, update_user_use_case: UpdateUser):
        self.add_user_use_case = add_user_use_case
        self.remove_user_use_case = remove_user_use_case
        self.update_user_use_case = update_user_use_case

    def add_user(self, user_data: dict) -> None:
        user = User(**user_data)
        self.add_user_use_case.execute(user)

    def remove_user(self, user_id: str) -> None:
        self.remove_user_use_case.execute(user_id)

    def update_user(self, user_data: dict) -> None:
        user = User(**user_data)
        self.update_user_use_case.execute(user)