"""Authentication failures do not disclose token details."""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from routes.ingest import router
from routes.documents import router as documents_router
from routes.grounding import make_grounding_router
from config import GeminiSettings


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_openapi_declares_shared_secret_bearer_auth():
    app = FastAPI()
    app.include_router(router)
    app.include_router(documents_router)
    app.include_router(make_grounding_router("test-key", settings=GeminiSettings()))
    schema = app.openapi()
    scheme = schema["components"]["securitySchemes"]["SharedSecretBearer"]
    assert scheme["type"] == "http"
    assert scheme["scheme"] == "bearer"
    for path, method in (
        ("/ingest", "post"), ("/documents/{document_id}", "delete"),
        ("/explain-with-grounding", "post"),
    ):
        assert schema["paths"][path][method]["security"] == [{"SharedSecretBearer": []}]


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


@pytest.mark.anyio
@pytest.mark.parametrize("method,path", [("POST", "/ingest"), ("DELETE", "/documents/test")])
@pytest.mark.parametrize("missing_database", [False, True])
async def test_ingestion_routes_retain_readiness_checks(monkeypatch, method, path, missing_database):
    from auth import UNAVAILABLE, UNCONFIGURED

    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-secret")
    if missing_database:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", "postgresql://test.invalid/test")
    app = FastAPI()
    app.state.ingest_status = "disabled (db unreachable)"
    database = AsyncMock()
    app.state.ingest_database = database
    app.include_router(router)
    app.include_router(documents_router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.request(
            method, path, json={} if method == "POST" else None,
            headers={"Authorization": "Bearer test-secret"},
        )
    assert response.status_code == 503
    assert response.json() == {"detail": UNCONFIGURED if missing_database else UNAVAILABLE}
    database.sessions.assert_not_called()
