"""Authentication failures do not disclose token details."""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from routes.ingest import router


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic wrong"])
async def test_auth_rejects_missing_or_wrong_token(monkeypatch, header):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test.invalid/test")
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-secret")
    app = FastAPI()
    app.state.ingest_status = "ready"
    app.state.ingest_database = AsyncMock()
    app.include_router(router)
    headers = {} if header is None else {"Authorization": header}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ingest", json={}, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
