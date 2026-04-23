from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    databricks_host: str | None
    databricks_token: str | None
    databricks_llm_endpoint: str | None
    databricks_llm_enabled: bool


DEFAULT_DATABASE_URL = "postgresql+psycopg://panda:panda@localhost:5432/panda_matching"


def get_settings() -> Settings:
    llm_enabled_raw = os.getenv("DATABRICKS_LLM_ENABLED", "false").strip().lower()
    llm_enabled = llm_enabled_raw in {"1", "true", "yes", "on"}
    return Settings(
        database_url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        databricks_host=os.getenv("DATABRICKS_HOST"),
        databricks_token=os.getenv("DATABRICKS_TOKEN"),
        databricks_llm_endpoint=os.getenv("DATABRICKS_LLM_ENDPOINT"),
        databricks_llm_enabled=llm_enabled,
    )
