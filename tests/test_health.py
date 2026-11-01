"""Startup readiness after transient database probe failures."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from fastapi import FastAPI

from config import GeminiSettings


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_startup_database_probe_succeeds_on_third_attempt(monkeypatch, caplog):
    import main

    monkeypatch.setenv("DATABASE_URL", "postgresql://test.invalid/test")
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-inbound")
    monkeypatch.setenv("AI_SERVICE_CALLBACK_SECRET", "test-callback")
    monkeypatch.setenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", "example.com")
    monkeypatch.setenv("WORKER_COUNT", "1")
    monkeypatch.setattr(main, "gemini_settings", GeminiSettings(
        llm_provider="gemini_enterprise", google_cloud_project="test"
    ))
    database = SimpleNamespace(dispose=AsyncMock())
    probe = AsyncMock(side_effect=[TimeoutError(), OSError(), None])
    sleep = AsyncMock()
    wait_for = AsyncMock(wraps=asyncio.wait_for)
    recover = AsyncMock()
    hold = asyncio.Event()

    async def worker(*args):
        await hold.wait()

    monkeypatch.setattr(main, "Database", Mock(return_value=database))
    monkeypatch.setattr(main, "probe_ingestion_database", probe)
    monkeypatch.setattr(main.asyncio, "sleep", sleep)
    monkeypatch.setattr(main.asyncio, "wait_for", wait_for)
    monkeypatch.setattr(main, "recover_stuck_jobs", recover)
    monkeypatch.setattr(main, "worker_loop", worker)
    caplog.set_level(logging.INFO, logger=main.__name__)
    application = FastAPI()

    async with main.lifespan(application):
        assert application.state.ingest_status == "ready"
        assert application.state.worker_status == "running"
        assert probe.await_args_list == [call(database)] * 3
        assert [entry.kwargs for entry in wait_for.await_args_list] == [{"timeout": 20.0}] * 3
        assert sleep.await_args_list == [call(1), call(1)]
        recover.assert_awaited_once_with(database)
        assert [record.getMessage() for record in caplog.records if record.name == main.__name__] == [
            "Startup DB probe attempt 1/3: timeout",
            "Startup DB probe attempt 2/3: error",
            "Startup DB probe attempt 3/3: success",
        ]

    database.dispose.assert_awaited_once()
