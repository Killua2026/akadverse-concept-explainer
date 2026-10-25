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


@pytest.mark.parametrize("value, expected", [(None, 1), ("1", 1), ("3", 3)])
def test_worker_count_valid(monkeypatch, value, expected):
    from config import get_worker_count
    if value is None:
        monkeypatch.delenv("WORKER_COUNT", raising=False)
    else:
        monkeypatch.setenv("WORKER_COUNT", value)
    assert get_worker_count() == expected


@pytest.mark.parametrize("value", ["0", "-1", "abc", "1.5", ""])
def test_worker_count_invalid(monkeypatch, value):
    from config import get_worker_count
    monkeypatch.setenv("WORKER_COUNT", value)
    with pytest.raises(RuntimeError, match="^WORKER_COUNT must be an integer >= 1$"):
        get_worker_count()


def test_multiple_workers_no_collision(monkeypatch, caplog):
    from collections import Counter
    from pathlib import Path
    from uuid import uuid4
    from fastapi.testclient import TestClient
    from config import StorageSettings
    from db import Database
    import main

    schema = f"concept_explainer_test_{uuid4().hex}"
    monkeypatch.setenv("WORKER_COUNT", "3")
    monkeypatch.setattr(main, "gemini_settings", GeminiSettings(
        llm_provider="gemini_enterprise", google_cloud_project="test-project"
    ))
    monkeypatch.setattr(ingestion, "text", lambda sql: text(sql.replace("concept_explainer", schema)))
    caplog.set_level(logging.INFO)
    database = None
    claimed = set()
    all_claimed = asyncio.Event()
    original_claim = ingestion.claim_next_queued_job

    async def claim(db):
        job = await original_claim(db)
        if job is not None:
            claimed.add(job["document_id"])
            if len(claimed) == 3:
                all_claimed.set()
        return job

    async def download(url, path):
        # Hold all three jobs in processing to prove overlapping claims.
        await asyncio.wait_for(all_claimed.wait(), timeout=30)

    monkeypatch.setattr(ingestion, "claim_next_queued_job", claim)
    monkeypatch.setattr(ingestion, "download_file", download)
    monkeypatch.setattr(ingestion, "extract_pdf", lambda path: path.name.split("_")[-1])
    monkeypatch.setattr(ingestion, "RecursiveCharacterTextSplitter", Mock(return_value=SimpleNamespace(
        split_text=lambda content: [f"{content} chunk {i}" for i in range(int(content[0]))]
    )))
    monkeypatch.setattr(ingestion, "embed_chunks", AsyncMock(
        side_effect=lambda chunks, settings: [[0.5] * 768 for _ in chunks]
    ))

    async def configure(app):
        nonlocal database
        database = Database(StorageSettings.from_env())
        directory = Path(__file__).resolve().parents[1] / "migrations"
        try:
            async with database.engine.begin() as connection:
                for name in ("001_pgvector_storage.sql", "002_jobs_table.sql"):
                    migration = await asyncio.to_thread((directory / name).read_text, encoding="utf-8")
                    for statement in migration.replace("concept_explainer", schema).split(";"):
                        statement = statement.strip()
                        if statement and statement not in {"BEGIN", "COMMIT"}:
                            await connection.execute(text(statement))
                for index in range(1, 4):
                    payload = {"file_name": f"{index}.pdf", "file_url": f"https://example.com/{index}",
                               "mime_type": "application/pdf", "metadata": {"weeks": [index]}}
                    await connection.execute(text(f"INSERT INTO {schema}.jobs (document_id,payload) VALUES (:id,CAST(:payload AS jsonb))"),
                                             {"id": f"job-{index}", "payload": json.dumps(payload)})
        except BaseException:
            async with database.engine.begin() as connection:
                await connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
            await database.dispose()
            raise
        app.state.ingest_database = database
        app.state.ingest_status = "ready"

    monkeypatch.setattr(main, "configure_ingestion", configure)

    async def wait_for_completion():
        async def wait():
            while True:
                async with database.sessions() as session:
                    rows = (await session.execute(text(f"SELECT document_id,status,attempts FROM {schema}.jobs"))).all()
                if all(row.status == "done" for row in rows):
                    assert len(rows) == 3
                    assert all(row.attempts == 1 for row in rows)
                    return
                assert not any(row.status == "failed" for row in rows)
                await asyncio.sleep(0.05)
        await asyncio.wait_for(wait(), timeout=60)
        async with database.sessions() as session:
            counts = dict((await session.execute(text(f"SELECT document_id,count(*) FROM {schema}.chunks GROUP BY document_id"))).all())
        assert counts == {"job-1": 1, "job-2": 2, "job-3": 3}

    async def cleanup():
        async with database.engine.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))

    with TestClient(main.app) as client:
        try:
            assert client.get("/health").json()["worker_count"] == 3
            client.portal.call(wait_for_completion)
        finally:
            for task in main.app.state.worker_tasks:
                client.portal.call(task.cancel)
            async def settle():
                await asyncio.gather(*main.app.state.worker_tasks, return_exceptions=True)
            client.portal.call(settle)
            client.portal.call(cleanup)
    claims = Counter(record.document_id for record in caplog.records if record.getMessage() == "job.claimed")
    assert claims == {"job-1": 1, "job-2": 1, "job-3": 1}
    starts = [record.worker_index for record in caplog.records if record.getMessage() == "worker.started"]
    assert sorted(starts) == [0, 1, 2]


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
        assert main.app.state.worker_tasks == []
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
        assert len(main.app.state.worker_tasks) == 1
    assert all(task.done() for task in main.app.state.worker_tasks)
