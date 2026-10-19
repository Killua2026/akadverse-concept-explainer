"""Async Postgres resources, independent of the existing FastAPI app."""

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from config import StorageSettings


class Database:
    def __init__(self, settings: StorageSettings) -> None:
        try:
            url = make_url(settings.database_url.get_secret_value())
        except ArgumentError:
            raise RuntimeError("DATABASE_URL must be a valid Postgres URL.") from None
        if url.drivername not in {
            "postgres", "postgresql", "postgresql+psycopg"
        }:
            raise RuntimeError("DATABASE_URL must use Postgres with psycopg.")
        self.engine = create_async_engine(
            url.set(drivername="postgresql+psycopg"),
            poolclass=NullPool,
            hide_parameters=True,
            connect_args={"connect_timeout": 15, "prepare_threshold": None},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def dispose(self) -> None:
        await self.engine.dispose()
