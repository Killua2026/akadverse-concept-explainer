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
from services.callback import CallbackSettings
from db import Database
from workers import ingestion


@pytest.fixture(autouse=True)
def callback_mock(monkeypatch):
    callback = AsyncMock(return_value=True)
    monkeypatch.setattr(ingestion, "post_callback", callback)
    return callback


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
    import main

    schema = f"concept_explainer_test_{uuid4().hex}"
    monkeypatch.setenv("WORKER_COUNT", "3")
    monkeypatch.setattr(main, "gemini_settings", GeminiSettings(
        llm_provider="gemini_enterprise", google_cloud_project="test-project"
    ))
    monkeypatch.setattr(ingestion, "text", lambda sql: text(sql.replace("concept_explainer", schema)))
    caplog.set_level(logging.INFO)
    database: Database | None = None
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
    monkeypatch.setattr("services.extractors.pdf.extract_text", lambda path: path.name.split("_")[-1])
    monkeypatch.setattr("services.chunking.RecursiveCharacterTextSplitter", Mock(return_value=SimpleNamespace(
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
                               "callback_url": "https://example.com/callback", "mime_type": "application/pdf", "metadata": {"weeks": [index]}}
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
        assert database is not None
        async def wait():
            assert database is not None
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
        assert database is not None
        async with database.engine.begin() as connection:
            await connection.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))

    with TestClient(main.app) as client:
        portal = client.portal
        assert portal is not None
        try:
            assert client.get("/health").json()["worker_count"] == 3
            portal.call(wait_for_completion)
        finally:
            for task in main.app.state.worker_tasks:
                portal.call(task.cancel)
            async def settle():
                await asyncio.gather(*main.app.state.worker_tasks, return_exceptions=True)
            portal.call(settle)
            portal.call(cleanup)
    claims = Counter(record.document_id for record in caplog.records if record.getMessage() == "job.claimed")
    assert claims == {"job-1": 1, "job-2": 1, "job-3": 1}
    starts = [record.worker_index for record in caplog.records if record.getMessage() == "worker.started"]
    assert sorted(starts) == [0, 1, 2]


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def seed(database, schema, mime="application/pdf"):
    payload = {"mime_type": mime, "file_name": "notes.pdf", "file_url": "https://example.com/file", "callback_url": "https://example.com/callback", "metadata": {"weeks": [1]}}
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
        monkeypatch.setattr("services.chunking.RecursiveCharacterTextSplitter", Mock(return_value=SimpleNamespace(split_text=lambda value: [f"New content {i}" for i in range(count)])))
    return paths


def test_claim_marks_processing(worker_database):
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            job = await ingestion.claim_next_queued_job(database)
            assert job is not None
            assert job["status"] == "processing"
            assert job["attempts"] == 1
            assert await ingestion.claim_next_queued_job(database) is None
    asyncio.run(check())


def test_pdf_end_to_end(worker_database, monkeypatch, caplog, callback_mock):
    paths = mock_pdf(monkeypatch)
    caplog.set_level(logging.INFO)
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
            async with database.sessions() as session:
                assert await session.scalar(text(f"SELECT status FROM {schema}.jobs")) == "done"
                row = (await session.execute(text(f"SELECT content,metadata FROM {schema}.chunks"))).one()
                assert "New lecture content" in row.content
                assert row.metadata == {"weeks": [1]}
            assert not paths[0].exists()
    asyncio.run(check())
    assert callback_mock.await_args.kwargs["status"] == "ingested"
    assert callback_mock.await_args.kwargs["chunk_count"] == 1


def test_recovery_requeues_stale_processing(worker_database):
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            async with database.sessions() as session, session.begin():
                await session.execute(text(f"UPDATE {schema}.jobs SET status='processing', updated_at=now()-interval '6 minutes'"))
            await ingestion.recover_stuck_jobs(database)
            job = await ingestion.claim_next_queued_job(database)
            assert job is not None
            assert job["status"] == "processing"
            await ingestion.recover_stuck_jobs(database)
            assert await ingestion.claim_next_queued_job(database) is None
    asyncio.run(check())


def test_mid_job_failure_marks_failed(worker_database, monkeypatch):
    monkeypatch.setattr(ingestion, "download_file", AsyncMock(side_effect=RuntimeError("private")))
    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
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
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
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
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
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
        await ingestion.worker_loop(Mock(spec=Database), GeminiSettings(), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
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
        await ingestion.worker_loop(Mock(spec=Database), GeminiSettings(), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
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


def test_success_fires_callback_after_commit(worker_database, monkeypatch, callback_mock):
    mock_pdf(monkeypatch)

    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            client = Mock(spec=httpx.AsyncClient)
            settings = CallbackSettings.from_env()

            async def observe(shared_client, **body):
                assert shared_client is client
                assert body == {
                    "url": "https://example.com/callback", "secret": settings.callback_secret,
                    "document_id": "abc", "status": "ingested", "chunk_count": 1,
                }
                async with database.sessions() as session:
                    assert await session.scalar(text(f"SELECT status FROM {schema}.jobs")) == "done"
                    assert await session.scalar(text(f"SELECT count(*) FROM {schema}.chunks")) == 1
                return True

            callback_mock.side_effect = observe
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), client, settings)
            callback_mock.assert_awaited_once()
    asyncio.run(check())


@pytest.mark.parametrize("raw_error,expected", [
    ("Failure at /tmp/private/file", "Processing failed (see server logs)"),
    ("Extraction\n failed\t safely", "Extraction failed safely"),
    ("x" * 600, "x" * 500),
])
def test_failure_fires_callback_with_sanitized_error(worker_database, monkeypatch, callback_mock, raw_error, expected):
    from services.extractors import ExtractionError
    mock_pdf(monkeypatch)
    monkeypatch.setattr(ingestion, "extract_text", Mock(side_effect=ExtractionError(raw_error)))

    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)

            async def observe(client, **body):
                assert body["status"] == "failed"
                assert body["error"] == expected
                assert "chunk_count" not in body
                async with database.sessions() as session:
                    row = (await session.execute(text(f"SELECT status,last_error FROM {schema}.jobs"))).one()
                    assert row.status == "failed" and row.last_error == body["error"]
                return True

            callback_mock.side_effect = observe
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
            callback_mock.assert_awaited_once()
    asyncio.run(check())


@pytest.mark.parametrize("failed", [False, True])
def test_callback_failure_does_not_change_job_status(worker_database, monkeypatch, callback_mock, failed):
    mock_pdf(monkeypatch)
    if failed:
        monkeypatch.setattr(ingestion, "download_file", AsyncMock(side_effect=RuntimeError("simulated")))
    callback_mock.return_value = False

    async def check():
        async with worker_database() as (database, schema):
            await seed(database, schema)
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
            async with database.sessions() as session:
                assert await session.scalar(text(f"SELECT status FROM {schema}.jobs")) == ("failed" if failed else "done")
                assert await session.scalar(text(f"SELECT count(*) FROM {schema}.chunks")) == (0 if failed else 1)
            callback_mock.assert_awaited_once()
    asyncio.run(check())


@pytest.mark.parametrize("fail_processing", [False, True])
def test_delete_during_processing_cancels_job(worker_database, monkeypatch, caplog, fail_processing, callback_mock):
    from fastapi import FastAPI
    from routes import documents

    mock_pdf(monkeypatch)
    original_download = ingestion.download_file
    original_process = ingestion.process_job
    caplog.set_level(logging.INFO)

    async def check():
        started = asyncio.Event()
        resume = asyncio.Event()
        finished = asyncio.Event()
        hold_after_processing = asyncio.Event()

        async def download(url, path):
            await original_download(url, path)
            started.set()
            await resume.wait()
            if fail_processing:
                raise RuntimeError("Test processing failure")

        async def process(*args):
            try:
                await original_process(*args)
            finally:
                finished.set()
                # Keep teardown cancellation away from the fixture's shared DB connection.
                await hold_after_processing.wait()

        monkeypatch.setattr(ingestion, "download_file", download)
        monkeypatch.setattr(ingestion, "process_job", process)
        async with worker_database() as (database, schema):
            await seed(database, schema)
            monkeypatch.setattr(documents, "text", lambda sql: text(sql.replace("concept_explainer", schema)))
            monkeypatch.setenv("DATABASE_URL", "configured")
            monkeypatch.setenv("AI_SERVICE_API_KEY", "test-inbound")
            app = FastAPI()
            app.include_router(documents.router)
            app.state.ingest_status = "ready"
            app.state.ingest_database = database
            task = asyncio.create_task(ingestion.worker_loop(database, GeminiSettings(), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env()))
            try:
                await asyncio.wait_for(started.wait(), timeout=10)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.delete("/documents/abc", headers={"Authorization": "Bearer test-inbound"})
                assert response.status_code == 204
                resume.set()
                await asyncio.wait_for(finished.wait(), timeout=10)
            finally:
                resume.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            assert task.done()
            async with database.sessions() as session:
                assert await session.scalar(text(f"SELECT count(*) FROM {schema}.chunks")) == 0
                assert await session.scalar(text(f"SELECT count(*) FROM {schema}.jobs")) == 0
    asyncio.run(check())
    assert any(record.message == "job.cancelled_during_processing" for record in caplog.records)
    callback_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_missing_callback_secret_disables_ingestion(monkeypatch):
    import main
    monkeypatch.setattr(main, "gemini_settings", GeminiSettings(
        llm_provider="gemini_enterprise", google_cloud_project="test"
    ))
    monkeypatch.setenv("DATABASE_URL", "configured")
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-inbound")
    monkeypatch.delenv("AI_SERVICE_CALLBACK_SECRET", raising=False)
    recover = AsyncMock()
    monkeypatch.setattr(main, "recover_stuck_jobs", recover)
    async with main.lifespan(main.app):
        assert main.app.state.ingest_status == "disabled (missing AI_SERVICE_CALLBACK_SECRET)"
        assert main.app.state.worker_tasks == []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            response = await client.post("/ingest", json={}, headers={"Authorization": "Bearer test-inbound"})
        assert response.status_code == 503
    recover.assert_not_awaited()


@pytest.mark.anyio
async def test_workers_share_callback_client_until_shutdown(monkeypatch):
    import main
    clients = []
    closed_during_cancellation = []
    started = asyncio.Event()
    hold = asyncio.Event()

    async def configure(app):
        app.state.ingest_status = "ready"
        app.state.ingest_database = SimpleNamespace(dispose=AsyncMock())

    async def worker(database, settings, client, callback_settings):
        clients.append(client)
        assert callback_settings is main.app.state.callback_settings
        if len(clients) == 3:
            started.set()
        try:
            await hold.wait()
        finally:
            closed_during_cancellation.append(client.is_closed)

    monkeypatch.setenv("WORKER_COUNT", "3")
    monkeypatch.setattr(main, "gemini_settings", GeminiSettings(
        llm_provider="gemini_enterprise", google_cloud_project="test"
    ))
    monkeypatch.setattr(main, "configure_ingestion", configure)
    monkeypatch.setattr(main, "recover_stuck_jobs", AsyncMock())
    monkeypatch.setattr(main, "worker_loop", worker)
    async with main.lifespan(main.app):
        await asyncio.wait_for(started.wait(), timeout=2)
        client = main.app.state.callback_client
        assert all(shared is client for shared in clients)
        assert not client.is_closed
    assert closed_during_cancellation == [False, False, False]
    assert client.is_closed
    assert all(task.done() for task in main.app.state.worker_tasks)
