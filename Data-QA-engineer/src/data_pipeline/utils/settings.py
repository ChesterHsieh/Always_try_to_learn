from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    app_name: str = Field(default="data-pipeline")
    environment: str = Field(default="dev")
    config_path: Path = Field(default=Path("configs/config.yaml"))

    class Config:
        env_prefix = "DP_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    def load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    load_dotenv()  # load .env when first accessed
    return AppSettings()
