"""Queue admission and isolated Postgres acceptance checks."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from routes.ingest import ENQUEUE_SQL, router


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def body():
    return {
        "document_id": "test-document", "file_url": "https://example.com/note.pdf",
        "file_name": "note.pdf", "mime_type": "application/pdf",
        "callback_url": "https://example.com/callback",
        "metadata": {
            "course_code": "CSC 201", "course_title": "Algorithms",
            "course_offering_id": "offering", "academic_session": "2025/2026",
            "semester": "First", "start_week": 3, "end_week": 5,
            "weeks": [3, 4, 5], "material_type": "NOTES",
            "title": "Notes", "document_id": "test-document",
        },
    }


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://test.invalid/test")
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-secret")
    session = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = session
    app = FastAPI()
    app.state.ingest_status = "ready"
    app.state.ingest_database = SimpleNamespace(sessions=Mock(return_value=context))
    app.include_router(router)
    return app, session


async def post(app, body, token="test-secret"):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/ingest", json=body, headers={"Authorization": f"Bearer {token}"})


@pytest.mark.anyio
async def test_ingest_returns_202(configured, body):
    app, session = configured
    response = await post(app, body)
    assert response.status_code == 202
    assert response.json() == {"status": "queued", "document_id": body["document_id"]}
    session.commit.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize("mime", ["application/zip", "text/plain"])
async def test_unsupported_mime_returns_422(configured, body, mime):
    app, session = configured
    body["mime_type"] = mime
    response = await post(app, body)
    assert response.status_code == 422
    assert response.json() == {"detail": "Unsupported file type. Supported: PDF, DOCX, PPTX."}
    session.execute.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["DATABASE_URL", "AI_SERVICE_API_KEY"])
async def test_missing_configuration_returns_503(configured, body, monkeypatch, name):
    app, _ = configured
    monkeypatch.delenv(name)
    response = await post(app, body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Ingestion is not configured on this service. Contact the AI team."}


@pytest.mark.anyio
async def test_auth_reads_token_per_request(configured, body, monkeypatch):
    app, _ = configured
    monkeypatch.setenv("AI_SERVICE_API_KEY", "replacement")
    assert (await post(app, body)).status_code == 401
    assert (await post(app, body, "replacement")).status_code == 202


@pytest.mark.anyio
async def test_runtime_database_failure(configured, body):
    app, session = configured
    session.execute.side_effect = OperationalError("enqueue", {}, RuntimeError())
    response = await post(app, body)
    assert response.status_code == 503
    assert response.json() == {"detail": "Ingestion is temporarily unavailable. Retry with backoff."}
    assert app.state.ingest_status == "disabled (db unreachable)"
    session.commit.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("missing, expected", [
    (None, "ready"), ("DATABASE_URL", "disabled (missing DATABASE_URL)"),
    ("AI_SERVICE_API_KEY", "disabled (missing AI_SERVICE_API_KEY)"),
])
async def test_health_ingest_status(configured, monkeypatch, missing, expected):
    import main

    if missing:
        monkeypatch.delenv(missing)
    database = SimpleNamespace(dispose=AsyncMock())
    monkeypatch.setattr(main, "Database", Mock(return_value=database))
    monkeypatch.setattr(main, "probe_ingestion_database", AsyncMock())
    await main.configure_ingestion(main.app)
    assert (await main.health_check())["ingest_status"] == expected


@pytest.mark.anyio
async def test_startup_probe_timeout(configured, monkeypatch):
    import main

    monkeypatch.setattr(main, "Database", Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(main, "probe_ingestion_database", AsyncMock(side_effect=TimeoutError))
    await main.configure_ingestion(main.app)
    assert main.app.state.ingest_status == "disabled (db unreachable)"


def test_queued_job_and_duplicate(storage_session, body):
    async def check():
        from models import IngestRequest

        async with storage_session() as (session, schema):
            query = text(ENQUEUE_SQL.replace("concept_explainer", schema))
            parameters = {"document_id": body["document_id"],
                          "payload": IngestRequest(**body).model_dump_json()}
            first = (await session.execute(query, parameters)).mappings().one()
            await session.commit()
            assert first["status"] == "queued"
            assert first["attempts"] == 0
            assert first["payload"] == body
            await session.execute(text(f"UPDATE {schema}.jobs SET status='done'"))
            second = (await session.execute(query, parameters)).mappings().one()
            await session.commit()
            assert second["id"] == first["id"]
            assert second["status"] == "queued"
            assert second["attempts"] == 1
            assert await session.scalar(text(f"SELECT count(*) FROM {schema}.jobs")) == 1

    asyncio.run(check())
