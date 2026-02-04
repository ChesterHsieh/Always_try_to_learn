from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    id: Optional[int]
    username: str
    email: str
    hashed_password: str
    is_active: bool = True

    def deactivate(self) -> None:
        self.is_active = False

    def activate(self) -> None:
        self.is_active = True

    def check_password(self, password: str) -> bool:
        # Placeholder for password check logic
        # In real scenarios, use a library like bcrypt to verify hashed passwords
        return self.hashed_password == password