"""Real isolated storage with mocked external I/O."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest
from pypdf import PdfWriter
from sqlalchemy import text

from config import GeminiSettings
from workers import ingestion


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def seed(database, schema, mime="application/pdf"):
    payload = {"mime_type": mime, "file_name": "notes.pdf", "file_url": "https://example.com/file", "metadata": {"weeks": [1]}}
    async with database.sessions() as session, session.begin():
        await session.execute(text(f"INSERT INTO {schema}.jobs (document_id,payload) VALUES ('abc',CAST(:payload AS jsonb))"), {"payload": json.dumps(payload)})


def mock_pdf(monkeypatch, count=1):
    paths = []
    # Construct a minimal PDF with real extractable text and no extra dependency.
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    async def download(url, path):
        paths.append(path)
        def write():
            writer = PdfWriter()
            page = writer.add_blank_page(width=300, height=300)
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 10 100 Td (New lecture content) Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
            with path.open("wb") as handle:
                writer.write(handle)
        await asyncio.to_thread(write)
    monkeypatch.setattr(ingestion, "download_file", download)
    monkeypatch.setattr(ingestion, "embed_chunks", AsyncMock(side_effect=lambda chunks, settings: [[0.5] * 768 for _ in chunks]))
    if count > 1:
        monkeypatch.setattr(ingestion, "RecursiveCharacterTextSplitter", Mock(return_value=SimpleNamespace(split_text=lambda value: [f"New content {i}" for i in range(count)])))
    return paths


def test_claim_marks_processing(worker_database):
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            job = await ingestion.claim_next_queued_job(database)
            assert job["status"] == "processing"
            assert job["attempts"] == 1
            assert await ingestion.claim_next_queued_job(database) is None
    asyncio.run(check())


def test_pdf_end_to_end(worker_database, monkeypatch, caplog):
    paths = mock_pdf(monkeypatch)
    caplog.set_level(logging.INFO)
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database))
            async with database.sessions() as session:
                assert await session.scalar(text(f"SELECT status FROM {schema}.jobs")) == "done"
                row = (await session.execute(text(f"SELECT content,metadata FROM {schema}.chunks"))).one()
                assert "New lecture content" in row.content
                assert row.metadata == {"weeks": [1]}
            assert not paths[0].exists()
    asyncio.run(check())
    assert any(r.message == "callback.would_fire" and r.status == "ingested" for r in caplog.records)


@pytest.mark.parametrize("mime", ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.openxmlformats-officedocument.presentationml.presentation"])
def test_docx_pptx_fail(worker_database, mime):
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema, mime)
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database))
            async with database.sessions() as session:
                row = (await session.execute(text(f"SELECT status,last_error FROM {schema}.jobs"))).one()
                assert row.status == "failed"
                assert row.last_error == ingestion.OFFICE_ERROR
    asyncio.run(check())


def test_recovery_requeues_stale_processing(worker_database):
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            async with database.sessions() as session, session.begin():
                await session.execute(text(f"UPDATE {schema}.jobs SET status='processing', updated_at=now()-interval '6 minutes'"))
            await ingestion.recover_stuck_jobs(database)
            assert (await ingestion.claim_next_queued_job(database))["status"] == "processing"
            await ingestion.recover_stuck_jobs(database)
            assert await ingestion.claim_next_queued_job(database) is None
    asyncio.run(check())


def test_mid_job_failure_marks_failed(worker_database, monkeypatch):
    monkeypatch.setattr(ingestion, "download_file", AsyncMock(side_effect=RuntimeError("private")))
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database))
            async with database.sessions() as session:
                row = (await session.execute(text(f"SELECT status,last_error FROM {schema}.jobs"))).one()
                assert row.status == "failed"
                assert row.last_error == "Download failed. Retry needed."
    asyncio.run(check())


def test_rerun_replaces_chunks(worker_database, monkeypatch):
    mock_pdf(monkeypatch, count=5)
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            async with database.sessions() as session, session.begin():
                await session.execute(text(f"UPDATE {schema}.jobs SET status='done'"))
                for i in range(3):
                    await session.execute(text(f"INSERT INTO {schema}.chunks (document_id,content,embedding,metadata) VALUES ('abc',:content,CAST(:vector AS vector),'{{}}'::jsonb)"), {"content": f"Old content {i}", "vector": json.dumps([0.5]*768)})
                await session.execute(text(f"UPDATE {schema}.jobs SET status='queued' WHERE document_id='abc'"))
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database))
            async with database.sessions() as session:
                rows = (await session.execute(text(f"SELECT content FROM {schema}.chunks WHERE document_id='abc'"))).scalars().all()
                assert len(rows) == 5
                assert set(rows) == {f"New content {i}" for i in range(5)}
                assert not any("Old content" in row for row in rows)
    asyncio.run(check())


def test_insert_failure_preserves_previous_chunks(worker_database, monkeypatch):
    mock_pdf(monkeypatch)
    monkeypatch.setattr(ingestion, "embed_chunks", AsyncMock(return_value=[[0.5]]))
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            async with database.sessions() as session, session.begin():
                await session.execute(text(f"INSERT INTO {schema}.chunks (document_id,content,embedding,metadata) VALUES ('abc','Old content',CAST(:vector AS vector),'{{}}'::jsonb)"), {"vector": json.dumps([0.5]*768)})
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database))
            async with database.sessions() as session:
                assert await session.scalar(text(f"SELECT status FROM {schema}.jobs")) == "failed"
                assert (await session.execute(text(f"SELECT content FROM {schema}.chunks"))).scalars().all() == ["Old content"]
    asyncio.run(check())


@pytest.mark.anyio
async def test_db_failure_stops_worker(monkeypatch):
    from sqlalchemy.exc import OperationalError
    claim = AsyncMock(side_effect=OperationalError("claim", {}, RuntimeError()))
    monkeypatch.setattr(ingestion, "claim_next_queued_job", claim)
    with pytest.raises(OperationalError):
        await ingestion.worker_loop(None, GeminiSettings())
    assert claim.await_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize("status, expected", [(403, "Signed URL expired"), (500, "Server error")])
async def test_download_http_errors(monkeypatch, tmp_path, status, expected):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status, request=request))
    monkeypatch.setattr(ingestion.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    with pytest.raises((ValueError, httpx.HTTPStatusError), match=expected):
        await ingestion.download_file("https://example.com/file", tmp_path / "file.pdf")


@pytest.mark.anyio
async def test_empty_queue_sleeps(monkeypatch):
    monkeypatch.setattr(ingestion, "claim_next_queued_job", AsyncMock(return_value=None))
    sleep = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(ingestion.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await ingestion.worker_loop(None, GeminiSettings())
    assert sleep.await_args_list == [call(2)]


@pytest.mark.anyio
async def test_developer_mode_does_not_start_worker(monkeypatch):
    import main
    monkeypatch.setattr(main, "gemini_settings", GeminiSettings(google_api_key="test"))
    async with main.lifespan(main.app):
        assert main.app.state.worker_task is None
        assert (await main.health_check())["worker_status"] == "disabled (ingestion requires enterprise mode)"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            response = await client.post("/ingest", json={})
        assert response.status_code == 503
        assert response.json() == {"detail": "Ingestion requires LLM_PROVIDER=gemini_enterprise. Developer API mode is not supported for ingestion."}


@pytest.mark.anyio
async def test_health_worker_running(monkeypatch):
    import main
    async def configure(app):
        app.state.ingest_status = "ready"
        app.state.ingest_database = SimpleNamespace(dispose=AsyncMock())
    loop = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(main, "gemini_settings", GeminiSettings(llm_provider="gemini_enterprise", google_cloud_project="test"))
    monkeypatch.setattr(main, "configure_ingestion", configure)
    monkeypatch.setattr(main, "recover_stuck_jobs", AsyncMock())
    monkeypatch.setattr(main, "worker_loop", loop)
    async with main.lifespan(main.app):
        assert (await main.health_check())["worker_status"] == "running"
        assert main.app.state.worker_task is not None
    assert main.app.state.worker_task.done()
