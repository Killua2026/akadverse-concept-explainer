"""Document deletion against isolated storage, with production auth."""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text

from routes import documents


@pytest.mark.parametrize("chunks,jobs", [(True, True), (True, False), (False, True), (False, False)])
def test_delete_document(worker_database, monkeypatch, chunks, jobs):
    async def check():
        async with worker_database() as (database, schema):
            monkeypatch.setattr(documents, "text", lambda sql: text(sql.replace("concept_explainer", schema)))
            monkeypatch.setenv("DATABASE_URL", "configured-for-test")
            monkeypatch.setenv("AI_SERVICE_API_KEY", "test-inbound")
            app = FastAPI()
            app.include_router(documents.router)
            app.state.ingest_status = "ready"
            app.state.ingest_database = database
            async with database.sessions() as session, session.begin():
                for document_id in ("other", "target"):
                    if document_id == "other" or chunks:
                        await session.execute(text(
                            f"INSERT INTO {schema}.chunks (document_id,content,embedding,metadata) "
                            "VALUES (:id,'content',CAST(:vector AS vector),'{}'::jsonb)"
                        ), {"id": document_id, "vector": json.dumps([0.5] * 768)})
                    if document_id == "other" or jobs:
                        await session.execute(text(
                            f"INSERT INTO {schema}.jobs (document_id,payload) VALUES (:id,'{{}}'::jsonb)"
                        ), {"id": document_id})
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                response = await client.delete("/documents/target", headers={"Authorization": "Bearer test-inbound"})
            assert response.status_code == (204 if chunks or jobs else 404)
            if chunks or jobs:
                assert response.content == b""
            else:
                assert response.json() == {"detail": "Document not found"}
            async with database.sessions() as session:
                for table in ("chunks", "jobs"):
                    ids = (await session.execute(text(f"SELECT document_id FROM {schema}.{table}"))).scalars().all()
                    assert ids == ["other"]
    asyncio.run(check())


@pytest.mark.parametrize("authorization", [None, "Bearer wrong"])
def test_delete_requires_auth(monkeypatch, authorization):
    monkeypatch.setenv("DATABASE_URL", "configured-for-test")
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-inbound")
    app = FastAPI()
    app.include_router(documents.router)
    app.state.ingest_status = "ready"

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/documents/target", headers={"Authorization": authorization} if authorization else {})
        assert response.status_code == 401
    asyncio.run(check())
