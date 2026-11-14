"""Lesson planning against isolated pgvector storage with mocked Gemini."""

import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text

from config import GeminiSettings
from routes.lesson_planning import make_lesson_planning_router
from services import lesson_planning


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-secret")
    settings = GeminiSettings(llm_provider="gemini_enterprise", google_cloud_project="test", model_names=["test-model"])
    generate = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    create_client = Mock(return_value=SimpleNamespace(aio=context))
    monkeypatch.setattr(lesson_planning, "create_client", create_client)

    app = FastAPI()
    app.include_router(make_lesson_planning_router(settings))
    app.state.ingest_database = None
    return SimpleNamespace(app=app, settings=settings, generate=generate, create_client=create_client)


@pytest.fixture
def lesson_database(worker_database, monkeypatch, configured):
    @asynccontextmanager
    async def connect():
        async with worker_database() as (database, schema):
            monkeypatch.setattr(lesson_planning, "text", lambda sql: text(sql.replace("concept_explainer", schema)))
            configured.app.state.ingest_database = database
            yield database, schema

    return connect


async def seed_chunks(database, schema, document_id, contents):
    chunk_ids = []
    async with database.sessions.begin() as session:
        for content in contents:
            chunk_id = await session.scalar(text(f"""
                INSERT INTO {schema}.chunks (document_id, content, embedding, metadata)
                VALUES (:document_id, :content, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                RETURNING id
            """), {
                "document_id": document_id,
                "content": content,
                "embedding": json.dumps([1] + [0] * 767),
                "metadata": json.dumps({"title": "Circuit Analysis"}),
            })
            chunk_ids.append(str(chunk_id))
    return chunk_ids


async def post(app, document_id, token="test-secret", **body):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/analyse-material", json={"document_id": document_id, **body}, headers=headers)


@pytest.mark.anyio
async def test_analyse_returns_structured_plan(configured, lesson_database):
    document_id = uuid4().hex
    topics = ["Ohm's law", "Series circuits", "Voltage division"]
    concept_ids = ["concept-1", "concept-2", "concept-3"]
    async with lesson_database() as (database, schema):
        chunk_ids = await seed_chunks(database, schema, document_id, topics)
        plan = {
            "sections": [{"section_id": "section-1", "title": "Circuit foundations", "concept_ids": concept_ids}],
            "concepts": [
                {
                    "concept_id": concept_id,
                    "title": topic,
                    "description": f"Introduce {topic}.",
                    "learning_objective": f"Apply {topic} in a circuit.",
                    "difficulty": "introductory",
                    "prerequisite_concept_ids": concept_ids[max(0, index - 1):index],
                    "source_chunk_ids": [chunk_id],
                }
                for index, (concept_id, topic, chunk_id) in enumerate(zip(concept_ids, topics, chunk_ids))
            ],
            "missing_prerequisites": [],
            "notes": None,
        }
        configured.generate.return_value = SimpleNamespace(text=json.dumps(plan), parsed=plan)
        response = await post(configured.app, document_id, max_concepts=3)

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "lesson_plan": {"document_id": document_id, **plan},
        "chunks_analysed": 3,
    }
    returned_plan = response.json()["lesson_plan"]
    assert returned_plan["sections"] and returned_plan["concepts"]
    earlier_ids = set()
    for concept in returned_plan["concepts"]:
        assert earlier_ids.issuperset(concept["prerequisite_concept_ids"])
        earlier_ids.add(concept["concept_id"])
    for section in returned_plan["sections"]:
        assert earlier_ids.issuperset(section["concept_ids"])

    configured.create_client.assert_called_once_with(configured.settings)
    configured.generate.assert_awaited_once()
    kwargs = configured.generate.await_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["config"].temperature == 0.2
    assert kwargs["config"].response_mime_type == "application/json"
    schema = kwargs["config"].response_schema
    assert "document_id" not in schema["properties"]
    assert "document_id" not in schema["required"]
    assert lesson_planning.TEXT_FALLBACK_SCHEMA not in kwargs["contents"]
    chunks = "\n\n".join(f"--- chunk {chunk_id} ---\n{topic}" for chunk_id, topic in zip(chunk_ids, topics))
    assert chunks in kwargs["contents"]


@pytest.mark.anyio
async def test_analyse_document_not_found_returns_404(configured, lesson_database):
    async with lesson_database():
        response = await post(configured.app, uuid4().hex)
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found or has no ingested chunks"}
    configured.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_analyse_malformed_llm_returns_fallback_plan(configured, lesson_database, caplog):
    document_id = uuid4().hex
    async with lesson_database() as (database, schema):
        await seed_chunks(database, schema, document_id, ["Ohm's law"])
        for invalid_response, reason in (
            ("garbage", "no_json_object"),
            ('{"concepts": []}', "invalid_concept_or_section_lists"),
        ):
            caplog.clear()
            configured.generate.return_value = SimpleNamespace(text=invalid_response)
            response = await post(configured.app, document_id)

            assert response.status_code == 200
            body = response.json()
            assert body["chunks_analysed"] == 1
            plan = body["lesson_plan"]
            assert len(plan["sections"]) == len(plan["concepts"]) == 1
            concept = plan["concepts"][0]
            assert plan["sections"][0]["concept_ids"] == [concept["concept_id"]]
            assert concept["title"] == "Overview of Circuit Analysis"
            assert concept["description"] == "The material could not be decomposed into a structured plan"
            assert any(
                record.levelno == logging.WARNING
                and record.getMessage().startswith(f"lesson_plan.fallback document_id={document_id} error_type=")
                and f" reason={reason} parsed=False finish_reason=None " in record.getMessage()
                and record.getMessage().endswith(f" raw_length={len(invalid_response)}")
                for record in caplog.records
            )


@pytest.mark.anyio
async def test_analyse_requires_auth(configured):
    for token in (None, "wrong"):
        response = await post(configured.app, "lecture", token=token)
        assert response.status_code == 401
    configured.generate.assert_not_awaited()


@pytest.mark.anyio
async def test_analyse_503_when_db_unavailable(configured):
    assert configured.app.state.ingest_database is None
    response = await post(configured.app, "lecture")
    assert response.status_code == 503
    configured.generate.assert_not_awaited()


@pytest.fixture
async def generated_plan(configured, lesson_database):
    document_id = uuid4().hex
    topics = ["Ohm's law", "Series circuits", "Voltage division"]
    concept_ids = ["concept-1", "concept-2", "concept-3"]
    async with lesson_database() as (database, schema):
        chunk_ids = await seed_chunks(database, schema, document_id, topics)
        plan = {
            "sections": [{"section_id": "section-1", "title": "Circuit foundations", "concept_ids": concept_ids}],
            "concepts": [
                {
                    "concept_id": concept_id,
                    "title": topic,
                    "description": f"Introduce {topic}.",
                    "learning_objective": f"Apply {topic} in a circuit.",
                    "difficulty": "introductory",
                    "prerequisite_concept_ids": concept_ids[:index],
                    "source_chunk_ids": [chunk_id],
                }
                for index, (concept_id, topic, chunk_id) in enumerate(zip(concept_ids, topics, chunk_ids))
            ],
        }
        yield document_id, plan


@pytest.mark.anyio
async def test_analyse_handles_fenced_json(configured, generated_plan):
    document_id, plan = generated_plan
    configured.generate.return_value = SimpleNamespace(text=f"```json\n{json.dumps(plan)}\n```")
    response = await post(configured.app, document_id)
    assert response.status_code == 200
    assert response.json()["lesson_plan"]["concepts"] == plan["concepts"]


@pytest.mark.anyio
async def test_analyse_handles_preamble_json(configured, generated_plan):
    document_id, plan = generated_plan
    configured.generate.return_value = SimpleNamespace(text=f"Here is the plan:\n{json.dumps(plan)}")
    response = await post(configured.app, document_id)
    assert response.status_code == 200
    assert response.json()["lesson_plan"]["concepts"] == plan["concepts"]


@pytest.mark.anyio
async def test_analyse_repairs_invalid_prerequisites(configured, generated_plan, caplog):
    document_id, plan = generated_plan
    plan["concepts"][1]["prerequisite_concept_ids"].extend(["unknown", "concept-2", "concept-3"])
    plan["sections"][0]["concept_ids"].append("unknown")
    plan["sections"].append({"section_id": "section-2", "title": "Empty", "concept_ids": ["unknown"]})
    configured.generate.return_value = SimpleNamespace(text=json.dumps(plan))
    response = await post(configured.app, document_id)
    assert response.status_code == 200
    repaired_plan = response.json()["lesson_plan"]
    assert len(repaired_plan["concepts"]) == 3
    assert repaired_plan["concepts"][1]["prerequisite_concept_ids"] == ["concept-1"]
    assert len(repaired_plan["sections"]) == 1
    assert repaired_plan["sections"][0]["concept_ids"] == ["concept-1", "concept-2", "concept-3"]
    assert f"lesson_plan.repaired document_id={document_id}" in caplog.messages
