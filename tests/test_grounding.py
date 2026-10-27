"""Grounding uses the existing ingestion shared-secret dependency."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from config import GeminiSettings
from routes import grounding


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def grounding_app(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-grounding-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused-test-database")
    app = FastAPI()
    app.state.ingest_status = "ready"
    app.include_router(grounding.make_grounding_router(
        "test-key", settings=GeminiSettings(google_api_key="test-key", model_names=["test-model"])
    ))
    return app


@pytest.mark.anyio
async def test_grounding_requires_auth(grounding_app, monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(grounding, "create_client", constructor)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=grounding_app), base_url="http://test"
    ) as client:
        response = await client.post("/explain-with-grounding", data={"question": "Recursion?"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    constructor.assert_not_called()


@pytest.mark.anyio
async def test_grounding_accepts_valid_auth(grounding_app, monkeypatch):
    fields = {name: "Test explanation" for name in (
        "concept", "simple_explanation", "analogy", "example", "worked_problem"
    )}
    generate = AsyncMock(return_value=SimpleNamespace(
        text=json.dumps(fields), candidates=[SimpleNamespace(grounding_metadata=None)]
    ))
    monkeypatch.setattr(grounding, "create_client", Mock(return_value=SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=grounding_app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/explain-with-grounding", data={"question": "Recursion?"},
            headers={"Authorization": "Bearer test-grounding-secret"},
        )
    assert response.status_code == 200
    assert response.json() == {**fields, "sources": []}
    generate.assert_awaited_once()
    assert generate.await_args.kwargs["model"] == "test-model"
