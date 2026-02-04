# src/core/use_cases/remove_user.py

class RemoveUser:
    def __init__(self, user_repository):
        self.user_repository = user_repository

    def execute(self, user_id):
        self.user_repository.remove_user(user_id)