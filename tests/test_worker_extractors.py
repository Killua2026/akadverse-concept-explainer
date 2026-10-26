"""Office extraction through real isolated worker database inserts."""

import asyncio
import json
import shutil
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import text

from config import GeminiSettings
import httpx
from services.callback import CallbackSettings
from services.extractors import ExtractionError
from workers import ingestion

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@pytest.fixture(autouse=True)
def callback_mock(monkeypatch):
    callback = AsyncMock(return_value=True)
    monkeypatch.setattr(ingestion, "post_callback", callback)
    return callback


@pytest.mark.parametrize("kind, mime", [("docx", DOCX_MIME), ("pptx", PPTX_MIME)])
def test_office_job_success(worker_database, monkeypatch, office_document, kind, mime, callback_mock):
    source = office_document(kind, paragraph="Lecture content")
    paths = []

    async def download(url, path):
        paths.append(path)
        await asyncio.to_thread(shutil.copyfile, source, path)

    monkeypatch.setattr(ingestion, "download_file", download)
    monkeypatch.setattr(ingestion, "embed_chunks", AsyncMock(return_value=[[0.5] * 768]))

    async def check():
        async with worker_database() as (database, schema):
            payload = {"file_name": source.name, "file_url": "https://example.com/file", "callback_url": "https://example.com/callback",
                       "mime_type": mime, "metadata": {"weeks": [1]}}
            async with database.sessions() as session, session.begin():
                await session.execute(text(f"INSERT INTO {schema}.jobs (document_id,payload) VALUES ('office',CAST(:payload AS jsonb))"),
                                      {"payload": json.dumps(payload)})
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
            async with database.sessions() as session:
                assert await session.scalar(text(f"SELECT status FROM {schema}.jobs")) == "done"
                row = (await session.execute(text(f"SELECT content,metadata FROM {schema}.chunks"))).one()
                assert row.content == "Lecture content"
                assert row.metadata == {"weeks": [1]}
            assert not paths[0].exists()
    asyncio.run(check())
    assert callback_mock.await_args.kwargs["status"] == "ingested"
    assert callback_mock.await_args.kwargs["chunk_count"] == 1


@pytest.mark.parametrize("message", ["No extractable text in DOCX", "Extraction failed: " + "x" * 600])
def test_extraction_error_marks_failed(worker_database, monkeypatch, message, callback_mock):
    monkeypatch.setattr(ingestion, "download_file", AsyncMock())
    extractor = AsyncMock()
    monkeypatch.setattr(ingestion, "embed_chunks", extractor)

    def fail(path, mime):
        raise ExtractionError(message)

    monkeypatch.setattr(ingestion, "extract_text", fail)

    async def check():
        async with worker_database() as (database, schema):
            payload = {"file_name": "empty.docx", "file_url": "https://example.com/file", "callback_url": "https://example.com/callback",
                       "mime_type": DOCX_MIME, "metadata": {}}
            async with database.sessions() as session, session.begin():
                await session.execute(text(f"INSERT INTO {schema}.jobs (document_id,payload) VALUES ('office',CAST(:payload AS jsonb))"),
                                      {"payload": json.dumps(payload)})
            await ingestion.process_job(database, GeminiSettings(), await ingestion.claim_next_queued_job(database), Mock(spec=httpx.AsyncClient), CallbackSettings.from_env())
            async with database.sessions() as session:
                row = (await session.execute(text(f"SELECT status,last_error FROM {schema}.jobs"))).one()
                assert row.status == "failed"
                assert row.last_error == message[:500]
            extractor.assert_not_awaited()
    asyncio.run(check())
    assert callback_mock.await_args.kwargs["status"] == "failed"
    assert callback_mock.await_args.kwargs["error"] == message[:500]
