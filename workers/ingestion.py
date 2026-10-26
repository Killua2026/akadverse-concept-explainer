"""Single-pass document ingestion with short database transactions."""

import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4

import httpx
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import GeminiSettings, WORKER_POLL_SECONDS
from db import Database
from services.embedding import embed_chunks
from services.chunking import chunk_text
from services.extractors import extract_text
from services.extractors.errors import ExtractionError

logger = logging.getLogger(__name__)
CLAIM_SQL = """UPDATE concept_explainer.jobs
SET status = 'processing', attempts = attempts + 1, updated_at = now()
WHERE id = (
  SELECT id FROM concept_explainer.jobs
  WHERE status = 'queued'
  ORDER BY created_at
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
RETURNING *;"""
RECOVERY_SQL = """UPDATE concept_explainer.jobs
SET status = 'queued', updated_at = now()
WHERE status = 'processing'
  AND updated_at < now() - interval '5 minutes';"""


async def recover_stuck_jobs(database: Database) -> None:
    async with database.sessions() as session, session.begin():
        await session.execute(text(RECOVERY_SQL))


async def claim_next_queued_job(database: Database):
    async with database.sessions() as session, session.begin():
        row = (await session.execute(text(CLAIM_SQL))).mappings().first()
        job = dict(row) if row else None
    return job


async def download_file(url: str, path: Path) -> None:
    async def stream():
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                if response.status_code == 403:
                    raise ValueError("Signed URL expired, retry needed")
                response.raise_for_status()
                size = 0
                handle = await asyncio.to_thread(path.open, "wb")
                try:
                    async for block in response.aiter_bytes():
                        size += len(block)
                        if size > 20 * 1024 * 1024:
                            raise ValueError("File exceeds the 20 MB limit.")
                        await asyncio.to_thread(handle.write, block)
                finally:
                    await asyncio.to_thread(handle.close)
    await asyncio.wait_for(stream(), timeout=60)


async def process_job(database: Database, settings: GeminiSettings, job) -> None:
    path = None
    stage = "Document validation"
    try:
        payload = job["payload"]
        filename = str(payload["file_name"]).replace("\\", "/").split("/")[-1]
        path = Path("/tmp") / f"{uuid4().hex}_{filename}"
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        stage = "Download"
        await download_file(payload["file_url"], path)
        stage = "Document extraction"
        content = await asyncio.to_thread(extract_text, path, payload["mime_type"])
        chunks = await asyncio.to_thread(chunk_text, content)
        stage = "Embedding"
        vectors = await embed_chunks(chunks, settings)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count mismatch")
        stage = "Database insert"
        async with database.sessions() as session, session.begin():
            await session.execute(text("SET LOCAL search_path TO public, extensions"))
            await session.execute(text("DELETE FROM concept_explainer.chunks WHERE document_id = :document_id"),
                                  {"document_id": job["document_id"]})
            for chunk, vector in zip(chunks, vectors, strict=True):
                await session.execute(text(
                    "INSERT INTO concept_explainer.chunks (document_id, content, embedding, metadata) "
                    "VALUES (:document_id, :content, CAST(:embedding AS vector), CAST(:metadata AS jsonb))"
                ), {"document_id": job["document_id"], "content": chunk,
                    "embedding": json.dumps(vector), "metadata": json.dumps(payload["metadata"])})
            await session.execute(text("UPDATE concept_explainer.jobs SET status='done', last_error=NULL, updated_at=now() WHERE id=:id"), {"id": job["id"]})
        logger.info("job.completed", extra={"document_id": job["document_id"]})
        logger.info("callback.would_fire", extra={"document_id": job["document_id"], "status": "ingested", "chunk_count": len(chunks)})
    except Exception as error:
        safe_messages = { "Signed URL expired, retry needed", "File exceeds the 20 MB limit."}
        message = str(error) if isinstance(error, ExtractionError) or (isinstance(error, ValueError) and str(error) in safe_messages) else f"{stage} failed. Retry needed."
        async with database.sessions() as session, session.begin():
            await session.execute(text("UPDATE concept_explainer.jobs SET status='failed', last_error=:error, updated_at=now() WHERE id=:id"), {"id": job["id"], "error": message[:500]})
        logger.warning("job.failed", extra={"document_id": job["document_id"]})
        logger.info("callback.would_fire", extra={"document_id": job["document_id"], "status": "failed", "error": message[:500]})
    finally:
        if path is not None:
            await asyncio.to_thread(path.unlink, missing_ok=True)


async def worker_loop(database: Database, settings: GeminiSettings) -> None:
    logger.info("worker.started")
    try:
        while True:
            job = await claim_next_queued_job(database)
            if job is None:
                await asyncio.sleep(WORKER_POLL_SECONDS)
                continue
            logger.info("job.claimed", extra={"document_id": job["document_id"]})
            await process_job(database, settings, job)
    except (SQLAlchemyError, OSError):
        logger.error("worker.db_unavailable")
        raise
    finally:
        logger.info("worker.stopped")
