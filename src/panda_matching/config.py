from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str


DEFAULT_DATABASE_URL = "postgresql+psycopg://panda:panda@localhost:5432/panda_matching"


def get_settings() -> Settings:
    return Settings(database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL))
