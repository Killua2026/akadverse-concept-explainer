"""Storage settings loaded explicitly from the process environment."""

#this prevents the need to set the environment variables in the system before running the application
from dotenv import load_dotenv
load_dotenv()

import os
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_LOCATION = "global"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH_SIZE = 10
EMBEDDING_BATCH_SLEEP_SECONDS = 8
WORKER_POLL_SECONDS = 2
STUCK_JOB_MINUTES = 5
from dataclasses import dataclass, field

from pydantic import SecretStr


@dataclass(frozen=True)
class StorageSettings:
    database_url: SecretStr = field(repr=False)

    @classmethod
    def from_env(cls) -> "StorageSettings":
        database_url = os.getenv("DATABASE_URL", "").strip()
        if not database_url:
            raise RuntimeError("DATABASE_URL environment variable is not set.")
        return cls(database_url=SecretStr(database_url))


@dataclass(frozen=True)
class GeminiSettings:
    google_api_key: str | None = field(default=None, repr=False)
    google_cloud_project: str | None = None
    google_cloud_location: str = "global"
    llm_provider: str = "gemini"
    model_names: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "GeminiSettings":
        from services.gemini_provider import discover_enterprise_model_names
        from services.model_discovery import get_valid_model_name

        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or None
        provider = os.getenv("LLM_PROVIDER", "gemini").strip()
        settings = cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", "").strip() or None,
            google_cloud_project=project,
            google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "global").strip() or "global",
            llm_provider=provider,
        )
        if settings.use_enterprise and project:
            settings.model_names.extend(discover_enterprise_model_names(
                project, settings.google_cloud_location
            ))
        elif provider == "gemini" and settings.google_api_key:
            settings.model_names.append(get_valid_model_name(settings.google_api_key))
        return settings

    @property
    def use_enterprise(self) -> bool:
        return self.llm_provider == "gemini_enterprise"

    def validate(self) -> None:
        if self.llm_provider not in {"gemini", "gemini_enterprise"}:
            raise RuntimeError("LLM_PROVIDER must be gemini or gemini_enterprise.")
        if self.use_enterprise and not self.google_cloud_project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is not set.")
        if not self.use_enterprise and not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is not set. Set it before starting the service."
            )
