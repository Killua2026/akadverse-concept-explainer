"""Read-only document status against isolated storage, with production auth."""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE
from routes import documents


@pytest.fixture
def status_app(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-inbound")
    app = FastAPI()
    app.include_router(documents.router)
    app.state.ingest_status = "ready"
    return app


async def get_status(app, document_id="target", authorization="Bearer test-inbound"):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(
            f"/documents/{document_id}/status",
            headers={"Authorization": authorization} if authorization else {},
        )


async def insert_job(database, schema, *, status="queued", last_error=None, chunks=0):
    async with database.sessions() as session, session.begin():
        await session.execute(text(
            f"INSERT INTO {schema}.jobs (document_id,payload,status,last_error,attempts) "
            "VALUES ('target','{}'::jsonb,:status,:last_error,:attempts)"
        ), {"status": status, "last_error": last_error, "attempts": 0 if status == "queued" else 1})
        for index in range(chunks):
            await session.execute(text(
                f"INSERT INTO {schema}.chunks (document_id,content,embedding,metadata) "
                "VALUES ('target',:content,CAST(:vector AS vector),'{}'::jsonb)"
            ), {"content": f"chunk {index}", "vector": json.dumps([0.5] * 768)})


def bind_database(app, database, schema, monkeypatch):
    app.state.ingest_database = database
    monkeypatch.setattr(documents, "text", lambda sql: text(sql.replace("concept_explainer", schema)))


def test_status_returns_complete_payload(worker_database, monkeypatch, status_app):
    async def check():
        async with worker_database() as (database, schema):
            bind_database(status_app, database, schema, monkeypatch)
            await insert_job(database, schema)
            response = await get_status(status_app)
            assert response.status_code == 200
            payload = response.json()
            assert set(payload) == {
                "document_id", "status", "chunk_count", "attempts", "last_error", "created_at", "updated_at",
            }
            assert payload == {
                **payload, "document_id": "target", "status": "queued", "chunk_count": 0,
                "attempts": 0, "last_error": None,
            }
            for field in ("created_at", "updated_at"):
                assert datetime.fromisoformat(payload[field].replace("Z", "+00:00")).tzinfo is not None
    asyncio.run(check())


def test_status_after_chunks_inserted(worker_database, monkeypatch, status_app):
    async def check():
        async with worker_database() as (database, schema):
            bind_database(status_app, database, schema, monkeypatch)
            await insert_job(database, schema, status="done", chunks=3)
            response = await get_status(status_app)
            assert response.status_code == 200
            assert response.json()["chunk_count"] == 3
            assert response.json()["status"] == "done"
            assert response.json()["attempts"] == 1
    asyncio.run(check())


def test_status_returns_failed_with_last_error(worker_database, monkeypatch, status_app):
    async def check():
        async with worker_database() as (database, schema):
            bind_database(status_app, database, schema, monkeypatch)
            await insert_job(database, schema, status="failed", last_error="Extraction failed")
            response = await get_status(status_app)
            assert response.status_code == 200
            assert response.json()["status"] == "failed"
            assert response.json()["last_error"] == "Extraction failed"
    asyncio.run(check())


def test_status_returns_404_for_unknown_document(worker_database, monkeypatch, status_app):
    async def check():
        async with worker_database() as (database, schema):
            bind_database(status_app, database, schema, monkeypatch)
            response = await get_status(status_app, "unknown")
            assert response.status_code == 404
            assert response.json() == {"detail": "Document not found"}
    asyncio.run(check())


@pytest.mark.parametrize("authorization", [None, "Bearer wrong"])
def test_status_requires_auth(status_app, authorization, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "configured-for-test")
    response = asyncio.run(get_status(status_app, authorization=authorization))
    assert response.status_code == 401


@pytest.mark.parametrize("error_type", [SQLAlchemyError, OSError])
def test_status_503_when_database_unavailable(status_app, error_type, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "configured-for-test")
    status_app.state.ingest_database = SimpleNamespace(sessions=Mock(side_effect=error_type("unavailable")))
    response = asyncio.run(get_status(status_app))
    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}
