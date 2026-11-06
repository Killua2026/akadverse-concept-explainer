"""Real pgvector retrieval with deterministic embeddings and mocked generation."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker

from config import GeminiSettings
from routes.explain import make_explain_router
from services import retrieval


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-secret")
    settings = GeminiSettings(llm_provider="gemini_enterprise", google_cloud_project="test",
                              model_names=["test-model"])
    generate = AsyncMock(return_value=SimpleNamespace(candidates=[object()], text="A grounded answer."))
    api = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    context = AsyncMock()
    context.__aenter__.return_value = api
    client = Mock(return_value=SimpleNamespace(aio=context))
    embed = AsyncMock(return_value=[[1.0] + [0.0] * 767])
    monkeypatch.setattr(retrieval, "embed_chunks", embed)
    monkeypatch.setattr(retrieval, "create_client", client)
    app = FastAPI()
    app.include_router(make_explain_router(settings))
    app.state.ingest_database = None
    return SimpleNamespace(app=app, settings=settings, generate=generate, embed=embed, client=client)


@pytest.fixture
def seeded(storage_session, monkeypatch, configured):
    @asynccontextmanager
    async def seed(rows=None):
        async with storage_session() as (session, schema):
            monkeypatch.setattr(retrieval, "text", lambda sql: text(sql.replace("concept_explainer", schema)))

            @asynccontextmanager
            async def connection():
                yield session.bind

            database = SimpleNamespace(sessions=async_sessionmaker(
                session.bind, expire_on_commit=False, join_transaction_mode="create_savepoint"),
                engine=SimpleNamespace(connect=connection), dispose=AsyncMock())
            rows = rows if rows is not None else [
                ("exact", [1], "offering", [1, 3]),
                ("diagonal", [0.707, 0.707], "offering", [1]),
                ("half", [0.5] * 4, "offering", [1]),
                ("orthogonal", [0, 1], "offering", [1]),
                ("other-offering", [1], "other", [1]),
                ("other-week", [1], "offering", [2]),
            ]
            for document, vector, offering, weeks in rows:
                await session.execute(text(f"""
                    INSERT INTO {schema}.chunks (document_id, content, embedding, metadata)
                    VALUES (:document, :content, CAST(:vector AS vector), CAST(:metadata AS jsonb))
                """), {"document": document, "content": document + " course material " * 30,
                       "vector": json.dumps(vector + [0] * (768 - len(vector))),
                       "metadata": json.dumps({"course_offering_id": offering, "weeks": weeks,
                                               "start_week": 1, "end_week": 3,
                                               "title": document, "material_type": "NOTES"})})
            await session.commit()
            configured.app.state.ingest_database = database
            yield database
    return seed


async def post(app, *, question="Explain the topic", week=1, offering="offering", token="test-secret"):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/explain", json={"question": question, "week": week,
                                 "course_offering_id": offering},
                                 headers={"Authorization": f"Bearer {token}"} if token else {})


@pytest.mark.anyio
async def test_explain_valid_answer_with_sources(configured, seeded):
    async with seeded():
        response = await post(configured.app)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A grounded answer."
    assert [s["document_id"] for s in body["sources"]] == ["exact", "diagonal", "half"]
    assert body["sources"][0] == {"document_id": "exact", "title": "exact", "material_type": "NOTES",
                                  "chunk_preview": ("exact" + " course material " * 30)[:200]}
    assert body["retrieval_info"] == {"chunks_available": 4, "chunks_above_threshold": 3,
                                      "similarity_threshold": 0.4}
    configured.embed.assert_awaited_once_with(["Explain the topic"], configured.settings)
    configured.client.assert_called_once_with(configured.settings)
    kwargs = configured.generate.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["config"].temperature == 0.3
    context = "\n\n".join(f"--- Source: {name} ---\n{name}" + " course material " * 30
                            for name in ["exact", "diagonal", "half"])
    assert kwargs["contents"] == retrieval.EXPLAIN_PROMPT_TEMPLATE.format(context=context, question="Explain the topic")


@pytest.mark.anyio
async def test_explain_empty_week(configured, seeded):
    async with seeded():
        body = (await post(configured.app, week=9)).json()
    assert body == {"answer": "No course material has been ingested for Week 9 of this offering yet.",
                    "sources": [], "retrieval_info": {"chunks_available": 0, "chunks_above_threshold": 0,
                                                      "similarity_threshold": 0.4}}
    configured.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_explain_below_threshold(configured, seeded):
    async with seeded([("orthogonal", [0, 1], "offering", [1])]):
        body = (await post(configured.app)).json()
    assert body["answer"] == "The course material for Week 1 does not cover this question."
    assert body["sources"] == []
    assert body["retrieval_info"] == {"chunks_available": 1, "chunks_above_threshold": 0, "similarity_threshold": 0.4}
    configured.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_explain_filters_offering_and_week(configured, seeded):
    async with seeded():
        body = (await post(configured.app, week=3)).json()
    assert [s["document_id"] for s in body["sources"]] == ["exact"]
    assert body["retrieval_info"]["chunks_available"] == 1
    assert body["retrieval_info"]["chunks_above_threshold"] == 1


@pytest.mark.anyio
async def test_explain_orders_and_limits_top_k(configured, seeded):
    rows = [(str(i), [1, i / 20], "offering", [1]) for i in range(10)]
    async with seeded(rows):
        body = (await post(configured.app)).json()
    assert [s["document_id"] for s in body["sources"]] == [str(i) for i in range(8)]
    assert body["retrieval_info"]["chunks_above_threshold"] == 10


@pytest.mark.anyio
@pytest.mark.parametrize("token", [None, "wrong"])
async def test_explain_auth_failure(configured, token):
    assert (await post(configured.app, token=token)).status_code == 401
    configured.embed.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["missing_url", "missing_state", "query_error"])
async def test_explain_database_unavailable(configured, monkeypatch, failure):
    if failure == "missing_url":
        monkeypatch.delenv("DATABASE_URL", raising=False)
    if failure == "query_error":
        configured.app.state.ingest_database = object()
        monkeypatch.setattr(retrieval, "_query_chunks", AsyncMock(side_effect=SQLAlchemyError("private")))
    response = await post(configured.app)
    assert response.status_code == 503
    assert "private" not in response.text


@pytest.mark.anyio
async def test_explain_developer_mode(configured):
    app = FastAPI()
    app.include_router(make_explain_router(GeminiSettings()))
    app.state.ingest_database = object()
    assert (await post(app)).status_code == 503


@pytest.mark.anyio
@pytest.mark.parametrize("kwargs", [{"question": ""}, {"question": "x" * 4001}, {"week": 0}, {"offering": ""}])
async def test_explain_validation_failure(configured, kwargs):
    assert (await post(configured.app, **kwargs)).status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["embedding", "generation", "empty_candidates", "empty_text"])
async def test_explain_provider_failure(configured, seeded, failure):
    if failure == "embedding":
        configured.embed.side_effect = RuntimeError("private")
    elif failure == "generation":
        configured.generate.side_effect = RuntimeError("private")
    else:
        configured.generate.return_value = SimpleNamespace(
            candidates=[] if failure == "empty_candidates" else [object()], text="")
    async with seeded():
        response = await post(configured.app)
    assert response.status_code == (503 if failure == "embedding" else 200)
    assert "private" not in response.text
    if failure != "embedding":
        assert response.json()["answer"] == retrieval.FALLBACK_ANSWER


@pytest.mark.anyio
async def test_explain_available_when_callback_config_missing(configured, seeded, monkeypatch):
    import main

    monkeypatch.delenv("AI_SERVICE_CALLBACK_SECRET", raising=False)
    monkeypatch.delenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(main, "gemini_settings", configured.settings)
    recover = AsyncMock()
    monkeypatch.setattr(main, "sweep_expired_attachments", AsyncMock(return_value=0))
    monkeypatch.setattr(main, "recover_stuck_jobs", recover)
    async with seeded() as database:
        monkeypatch.setattr(main, "Database", Mock(return_value=database))
        app = configured.app
        app.router.lifespan_context = main.lifespan
        app.add_api_route("/health", main.health_check)
        monkeypatch.setattr(main, "app", app)
        async with app.router.lifespan_context(app):
            assert app.state.ingest_database is database
            response = await post(app)
            assert response.status_code == 200
            assert response.json()["answer"] == "A grounded answer."
            assert response.json()["sources"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                health = (await client.get("/health")).json()
            assert health["ingest_status"] == "disabled (missing AI_SERVICE_CALLBACK_SECRET)"
            assert app.state.worker_tasks == []
    recover.assert_not_awaited()


@pytest.mark.anyio
async def test_failed_startup_disposes_database(configured, monkeypatch):
    import main

    database = SimpleNamespace(dispose=AsyncMock())
    monkeypatch.setenv("DATABASE_URL", "postgresql://test.invalid/test")
    monkeypatch.setattr(main, "Database", Mock(return_value=database))
    probe = AsyncMock(side_effect=SQLAlchemyError("private"))
    monkeypatch.setattr(main, "probe_ingestion_database", probe)
    monkeypatch.setattr(main.asyncio, "sleep", AsyncMock())
    await main.configure_ingestion(configured.app)
    assert probe.await_count == 3
    database.dispose.assert_awaited_once()
    assert configured.app.state.ingest_database is None
    assert configured.app.state.ingest_status == "disabled (db unreachable)"


@pytest.mark.anyio
@pytest.mark.parametrize("result", ["answer", "empty_candidates", "empty_text", "error"])
async def test_generate_answer_resilience(configured, result):
    if result == "error":
        configured.generate.side_effect = RuntimeError("private")
    else:
        configured.generate.return_value = SimpleNamespace(
            candidates=[] if result == "empty_candidates" else [object()],
            text="An answer." if result == "answer" else "")
    answer = await retrieval._generate_answer("Explain", [
        {"metadata": {"title": "Notes"}, "content": "Material"}
    ], configured.settings)
    assert answer == ("An answer." if result == "answer" else retrieval.FALLBACK_ANSWER)
