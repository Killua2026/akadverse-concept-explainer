"""Attachment HTTP contracts and isolated Postgres storage lifecycle."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import ATTACHMENT_ALLOWED_MIME_TYPES, ATTACHMENT_MAX_SIZE_BYTES
from routes.attachments import make_attachments_router
from services import attachments

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "attachment-test-key")
    app = FastAPI()
    app.include_router(make_attachments_router())
    app.state.gemini_settings = SimpleNamespace(use_enterprise=True)
    app.state.ingest_database = None
    return app


@pytest.fixture
def stored(worker_database, monkeypatch, api):
    @asynccontextmanager
    async def context():
        async with worker_database() as (database, schema):
            monkeypatch.setattr(attachments, "text", lambda sql: text(sql.replace("concept_explainer", schema)))
            api.state.ingest_database = database
            yield database, schema
    return context


async def request(api, method="POST", path="/attachments", *, token="attachment-test-key", **kwargs):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as client:
        return await client.request(method, path, headers={"Authorization": f"Bearer {token}"} if token else {}, **kwargs)


async def upload(api, content=b"student document", filename="student.pdf", mime="application/pdf", **kwargs):
    return await request(api, files={"file": (filename, content, mime)}, **kwargs)


async def expire(database, schema, identifier):
    async with database.sessions() as session, session.begin():
        await session.execute(text(f"UPDATE {schema}.attachments SET expires_at = now() - interval '1 minute' WHERE id = :id"), {"id": identifier})


async def test_upload_pdf_returns_metadata(api, stored):
    async with stored():
        before = datetime.now(timezone.utc)
        response = await upload(api)
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"attachment_id", "filename", "mime_type", "size_bytes", "created_at", "expires_at"}
        assert body["filename"] == "student.pdf"
        assert body["mime_type"] == "application/pdf"
        assert body["size_bytes"] == len(b"student document")
        assert before + timedelta(hours=6) <= datetime.fromisoformat(body["expires_at"]) <= datetime.now(timezone.utc) + timedelta(hours=6)


@pytest.mark.parametrize("mime", sorted(ATTACHMENT_ALLOWED_MIME_TYPES))
async def test_upload_all_supported_mime_types(api, stored, mime):
    async with stored():
        response = await upload(api, mime=mime)
        assert response.status_code == 200
        assert response.json()["mime_type"] == mime


async def test_upload_unsupported_mime_returns_422(api, stored):
    async with stored():
        response = await upload(api, mime="image/gif")
        assert response.status_code == 422
        assert response.json() == {"detail": "Unsupported file type. Supported: PDF, DOCX, PPTX, TXT, MD, PNG, JPG, WebP."}


async def test_upload_oversized_returns_413(api, stored):
    async with stored():
        response = await upload(api, content=b"x" * (ATTACHMENT_MAX_SIZE_BYTES + 1))
        assert response.status_code == 413
        assert response.json() == {"detail": "File exceeds 4 MiB limit."}


async def test_upload_empty_file_returns_422(api, stored):
    async with stored():
        response = await upload(api, content=b"")
        assert response.status_code == 422
        assert response.json() == {"detail": "Empty file."}


async def test_upload_filename_sanitized(api, stored):
    async with stored() as (database, schema):
        response = await upload(api, filename="../../etc/passwd.pdf")
        assert response.status_code == 200
        assert response.json()["filename"] == "passwd.pdf"
        async with database.sessions() as session:
            assert await session.scalar(text(f"SELECT filename FROM {schema}.attachments")) == "passwd.pdf"


async def test_upload_requires_auth(api):
    for token in (None, "wrong"):
        assert (await upload(api, token=token)).status_code == 401


async def test_upload_503_when_database_unavailable(api):
    response = await upload(api)
    assert response.status_code == 503
    assert response.json() == {"detail": "Attachments require a configured database."}


async def test_upload_503_in_developer_mode(api):
    api.state.ingest_database = object()
    for settings in (SimpleNamespace(use_enterprise=False), None):
        api.state.gemini_settings = settings
        response = await upload(api)
        assert response.status_code == 503
        assert response.json() == {"detail": "Attachments require LLM_PROVIDER=gemini_enterprise."}
    del api.state.gemini_settings
    assert (await upload(api)).status_code == 503


async def test_get_existing_attachment_returns_metadata(api, stored, monkeypatch):
    async with stored():
        body = (await upload(api)).json()
        # API-key-only ownership: a newly configured valid key accesses the same ID.
        monkeypatch.setenv("AI_SERVICE_API_KEY", "rotated-test-key")
        response = await request(api, "GET", f"/attachments/{body['attachment_id']}", token="rotated-test-key")
        assert response.status_code == 200
        assert response.json() == body


async def test_get_expired_attachment_returns_404(api, stored):
    async with stored() as (database, schema):
        identifier = (await upload(api)).json()["attachment_id"]
        await expire(database, schema, identifier)
        response = await request(api, "GET", f"/attachments/{identifier}")
        assert response.status_code == 404
        assert response.json() == {"detail": "Attachment not found"}
        assert await attachments.get_attachment_bytes(database, identifier) is None


async def test_get_nonexistent_returns_404(api, stored):
    async with stored():
        response = await request(api, "GET", f"/attachments/{uuid4()}")
        assert response.status_code == 404
        assert response.json() == {"detail": "Attachment not found"}


async def test_get_requires_auth(api):
    for token in (None, "wrong"):
        assert (await request(api, "GET", f"/attachments/{uuid4()}", token=token)).status_code == 401


async def test_delete_existing_attachment_returns_204(api, stored, monkeypatch):
    async with stored() as (database, schema):
        identifier = (await upload(api)).json()["attachment_id"]
        monkeypatch.setenv("AI_SERVICE_API_KEY", "rotated-test-key")
        response = await request(api, "DELETE", f"/attachments/{identifier}", token="rotated-test-key")
        assert response.status_code == 204
        assert response.content == b""
        async with database.sessions() as session:
            assert await session.scalar(text(f"SELECT count(*) FROM {schema}.attachments")) == 0


async def test_delete_nonexistent_returns_404(api, stored):
    async with stored():
        response = await request(api, "DELETE", f"/attachments/{uuid4()}")
        assert response.status_code == 404
        assert response.json() == {"detail": "Attachment not found"}


async def test_delete_requires_auth(api):
    for token in (None, "wrong"):
        assert (await request(api, "DELETE", f"/attachments/{uuid4()}", token=token)).status_code == 401


async def test_get_attachment_bytes_returns_content(api, stored):
    async with stored() as (database, schema):
        identifier = (await upload(api, content=b"\x00\xffbinary")).json()["attachment_id"]
        assert await attachments.get_attachment_bytes(database, identifier) == (b"\x00\xffbinary", "student.pdf", "application/pdf")


async def test_sweep_removes_expired_rows(stored):
    async with stored() as (database, schema):
        expired = await attachments.store_attachment(database, filename="old.pdf", mime_type="application/pdf", content=b"old")
        fresh = await attachments.store_attachment(database, filename="fresh.pdf", mime_type="application/pdf", content=b"fresh")
        await expire(database, schema, expired.attachment_id)
        assert await attachments.sweep_expired_attachments(database) == 1
        async with database.sessions() as session:
            assert (await session.scalars(text(f"SELECT id::text FROM {schema}.attachments"))).all() == [fresh.attachment_id]


async def test_startup_sweep_runs(stored, monkeypatch):
    import main

    async with stored() as (database, schema):
        old = await attachments.store_attachment(database, filename="old.pdf", mime_type="application/pdf", content=b"old")
        await expire(database, schema, old.attachment_id)
        database.dispose = AsyncMock()

        async def configure(application):
            application.state.ingest_database = database
            application.state.ingest_status = "disabled (missing AI_SERVICE_CALLBACK_SECRET)"

        monkeypatch.setattr(main, "gemini_settings", SimpleNamespace(use_enterprise=True, validate=lambda: None))
        monkeypatch.setattr(main, "configure_ingestion", configure)
        app = FastAPI()
        async with main.lifespan(app):
            async with database.sessions() as session:
                assert await session.scalar(text(f"SELECT count(*) FROM {schema}.attachments")) == 0
            task = app.state.attachment_sweep_task
            assert not task.done()
        assert task.cancelled()
        database.dispose.assert_awaited_once()


async def test_upload_missing_sanitized_filename(api, stored):
    async with stored():
        response = await upload(api, filename=".")
        assert response.status_code == 422
        assert response.json() == {"detail": "Filename is required."}


@pytest.mark.parametrize("method", ["POST", "GET", "DELETE"])
async def test_storage_failure_returns_503(api, monkeypatch, method):
    from routes import attachments as routes

    api.state.ingest_database = object()
    service = {"POST": "store_attachment", "GET": "get_attachment_metadata", "DELETE": "delete_attachment"}[method]
    monkeypatch.setattr(routes, service, AsyncMock(side_effect=SQLAlchemyError("unavailable")))
    response = await upload(api) if method == "POST" else await request(api, method, f"/attachments/{uuid4()}")
    assert response.status_code == 503
    assert response.json() == {"detail": "Storage unavailable. Retry with backoff."}

@pytest.mark.parametrize("method", ["GET", "DELETE"])
async def test_read_delete_availability_checks(api, method):
    path = f"/attachments/{uuid4()}"
    response = await request(api, method, path)
    assert response.status_code == 503
    assert response.json() == {"detail": "Attachments require a configured database."}
    api.state.ingest_database = object()
    del api.state.gemini_settings
    response = await request(api, method, path)
    assert response.status_code == 503
    assert response.json() == {"detail": "Attachments require LLM_PROVIDER=gemini_enterprise."}


async def test_periodic_sweep_survives_failure(monkeypatch, caplog):
    import asyncio
    import main

    database = object()
    sweep = AsyncMock(side_effect=[RuntimeError("transient"), 1])
    calls = 0

    async def sleep(seconds):
        nonlocal calls
        assert seconds == 3600
        calls += 1
        if calls == 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(main.asyncio, "sleep", sleep)
    monkeypatch.setattr(main, "sweep_expired_attachments", sweep)
    with pytest.raises(asyncio.CancelledError):
        await main._periodic_attachment_sweep(database)
    assert sweep.await_count == 2
    assert "attachments.sweep_failed error_type=RuntimeError" in caplog.text
    assert "transient" not in caplog.text
