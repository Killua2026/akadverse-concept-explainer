"""Real Postgres acceptance test for the Phase 2a storage migration."""

import asyncio
import json

import pytest
from sqlalchemy import text

from config import StorageSettings


def test_config_fails_fast_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(
        RuntimeError, match=r"^DATABASE_URL environment variable is not set\.$"
    ):
        StorageSettings.from_env()


def test_pgvector_storage_round_trip(storage_session):
    async def check_storage():
        async with storage_session() as (session, schema):
            metadata = {"course_offering_id": "storage-smoke", "weeks": [3, 4]}
            embedding = "[" + ",".join(["1"] + ["0"] * 767) + "]"
            chunk_id = await session.scalar(text(
                f"INSERT INTO {schema}.chunks "
                "(document_id, content, embedding, metadata) "
                "VALUES (:document_id, :content, CAST(:embedding AS vector), "
                "CAST(:metadata AS jsonb)) RETURNING id"
            ), {
                "document_id": "storage-smoke",
                "content": "Synthetic storage smoke test.",
                "embedding": embedding,
                "metadata": json.dumps(metadata),
            })
            await session.commit()
            row = (await session.execute(text(
                f"SELECT document_id, content, metadata, vector_dims(embedding) AS dimensions, "
                f"created_at FROM {schema}.chunks WHERE id = :id"
            ), {"id": chunk_id})).mappings().one()
            assert row["document_id"] == "storage-smoke"
            assert row["content"] == "Synthetic storage smoke test."
            assert row["metadata"] == metadata
            assert row["dimensions"] == 768
            assert row["created_at"] is not None
            indexes = dict((await session.execute(text(
                "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = :schema "
                "AND tablename = 'chunks'"
            ), {"schema": schema})).all())
            assert "USING hnsw" in indexes["chunks_embedding_hnsw_idx"]
            assert "vector_cosine_ops" in indexes["chunks_embedding_hnsw_idx"]
            assert "(document_id)" in indexes["chunks_document_id_idx"]
            assert "course_offering_id" in indexes["chunks_course_offering_id_idx"]
            deleted_id = await session.scalar(text(
                f"DELETE FROM {schema}.chunks WHERE id = :id RETURNING id"
            ), {"id": chunk_id})
            assert deleted_id == chunk_id
            await session.commit()
            assert await session.scalar(text(
                f"SELECT count(*) FROM {schema}.chunks WHERE id = :id"
            ), {"id": chunk_id}) == 0

    asyncio.run(check_storage())
