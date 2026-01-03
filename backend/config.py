from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the backend service.

    Values are sourced from environment variables where possible so the
    same build can be used across dev/stage/prod without code changes.
    """

    app_name: str = Field("Hospital Operations Backend", env="APP_NAME")
    environment: str = Field("development", env="APP_ENV")
    debug: bool = Field(False, env="APP_DEBUG")

    # Comma-separated list of origins, e.g.
    #   http://localhost:5173,http://127.0.0.1:5173
    cors_origins_raw: str = Field(
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
        env="CORS_ORIGINS",
    )

    # SQLite database file for persisting monitoring snapshots.
    monitoring_db_path: str = Field("monitoring.db", env="MONITORING_DB_PATH")

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    class Config:
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance.

    Using lru_cache makes settings effectively a singleton while remaining
    easy to override in tests.
    """

    return Settings()  # type: ignore[call-arg]
