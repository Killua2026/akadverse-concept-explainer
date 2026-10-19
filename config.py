"""Storage settings loaded explicitly from the process environment."""

import os
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
