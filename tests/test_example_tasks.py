"""Worked-example task endpoint tests with isolated Postgres and mocked Gemini."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text

from config import GeminiSettings
from routes.example_tasks import make_example_tasks_router
from services import example_tasks

pytestmark = pytest.mark.anyio
EXAMPLE = "Given 12 volts and 4 ohms, divide voltage by resistance to get 3 amps."


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
    monkeypatch.setattr(example_tasks, "create_client", Mock(return_value=SimpleNamespace(aio=context)))
    app = FastAPI()
    app.include_router(make_example_tasks_router(settings))
    return SimpleNamespace(app=app, generate=generate)


@pytest.fixture
async def material(configured, worker_database, monkeypatch):
    async with worker_database() as (database, schema):
        monkeypatch.setattr(example_tasks, "text", lambda sql: text(sql.replace("concept_explainer", schema)))
        configured.app.state.ingest_database = database
        document_id = uuid4().hex
        async with database.sessions.begin() as session:
            chunk_id = await session.scalar(text(f"""
                INSERT INTO {schema}.chunks (document_id, content, embedding, metadata)
                VALUES (:document_id, :content, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                RETURNING id
            """), {
                "document_id": document_id,
                "content": EXAMPLE,
                "embedding": json.dumps([1] + [0] * 767),
                "metadata": "{}",
            })
        yield document_id, str(chunk_id)


def task_payload():
    return {"tasks": [
        {
            "task_id": f"task-{sequence}", "sequence": sequence, "prompt": prompt,
            "response_type": response_type, "expected_answer": answer,
        }
        for sequence, (prompt, response_type, answer) in enumerate([
            ("Identify the resistance.", "number", "4"),
            ("Calculate the current.", "number", "3"),
            ("Explain which relationship you used.", "text", "Ohm's law relates voltage, resistance and current."),
        ], 1)
    ]}


async def post(app, document_id, token="test-secret", **body):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/generate-example-tasks", headers=headers, json={
            "document_id": document_id, "concept_id": "ohms-law", "concept_description": "Apply Ohm's law", **body,
        })


async def test_generate_tasks_without_example(configured, material):
    document_id, chunk_id = material
    configured.generate.side_effect = [
        SimpleNamespace(parsed={"source": "LECTURE_NOTE", "text": EXAMPLE}),
        SimpleNamespace(parsed=task_payload()),
    ]

    response = await post(configured.app, document_id)

    assert response.status_code == 200
    body = response.json()
    assert body["concept_id"] == "ohms-law"
    assert body["example"] == {"source": "LECTURE_NOTE", "text": EXAMPLE, "source_chunk_ids": [chunk_id]}
    assert 3 <= len(body["tasks"]) <= 6
    assert [task["sequence"] for task in body["tasks"]] == list(range(1, len(body["tasks"]) + 1))
    first, second = [call.kwargs["config"] for call in configured.generate.await_args_list]
    assert first.response_mime_type == second.response_mime_type == "application/json"
    assert set(first.response_schema.model_fields) == {"source", "text"}
    assert set(second.response_schema.model_fields) == {"tasks"}


async def test_generate_tasks_with_example_text(configured, material, caplog):
    document_id, chunk_id = material
    configured.generate.return_value = SimpleNamespace(parsed=task_payload())
    supplied = "  " + EXAMPLE.upper().replace(" ", "\t\n") + "  "

    response = await post(configured.app, document_id, example_text=supplied)

    assert response.status_code == 200
    assert response.json()["example"] == {"source": "LECTURE_NOTE", "text": supplied, "source_chunk_ids": [chunk_id]}
    configured.generate.assert_awaited_once()
    assert set(configured.generate.await_args.kwargs["config"].response_schema.model_fields) == {"tasks"}

    configured.generate.reset_mock()
    response = await post(configured.app, document_id, example_text=" \t\n ")
    assert response.status_code == 200
    assert response.json()["example"]["source"] == "AI_GENERATED"
    assert response.json()["example"]["source_chunk_ids"] == []
    assert f"example_tasks.empty_example_text document_id={document_id}" in caplog.messages
    configured.generate.assert_awaited_once()


async def test_generate_tasks_repairs_bad_sequence(configured, material, caplog):
    document_id, _ = material
    payload = task_payload()
    for task, sequence in zip(payload["tasks"], [3, 1, 2]):
        task["sequence"] = sequence
    payload["tasks"].extend([
        payload["tasks"][0].copy(),
        {**payload["tasks"][0], "task_id": "bad-answer", "expected_answer": " "},
        {**payload["tasks"][0], "task_id": "bad-type", "response_type": "unknown"},
    ])
    configured.generate.side_effect = [
        RuntimeError("structured generation unavailable"),
        SimpleNamespace(text=f"```json\n{json.dumps(payload)}\n```"),
    ]

    response = await post(configured.app, document_id, example_text=EXAMPLE)

    assert response.status_code == 200
    tasks = response.json()["tasks"]
    assert [task["sequence"] for task in tasks] == [1, 2, 3]
    assert [task["task_id"] for task in tasks] == ["task-1", "task-2", "task-3"]
    assert configured.generate.await_count == 2
    assert configured.generate.await_args.kwargs["config"].response_schema is None
    assert f"example_tasks.repaired document_id={document_id}" in caplog.messages


async def test_generate_tasks_fallback_on_garbage(configured, material, caplog):
    document_id, _ = material
    configured.generate.return_value = SimpleNamespace(text="unparseable")

    response = await post(configured.app, document_id, example_text=EXAMPLE)

    assert response.status_code == 200
    body = response.json()
    assert body["example"] == {"source": "AI_GENERATED", "text": EXAMPLE, "source_chunk_ids": []}
    task, = body["tasks"]
    assert task["sequence"] == 1
    assert task["prompt"] == "Read the worked example and explain the first step."
    assert task["expected_answer"] == "A description of the first step in the example."
    assert any(message.startswith(f"example_tasks.fallback document_id={document_id} ") for message in caplog.messages)


async def test_generate_tasks_requires_auth(configured):
    for token in (None, "wrong"):
        response = await post(configured.app, uuid4().hex, token=token)
        assert response.status_code == 401
    configured.generate.assert_not_awaited()


async def test_generate_tasks_404_unknown_document(configured, material):
    response = await post(configured.app, uuid4().hex)
    assert response.status_code == 404
    configured.generate.assert_not_awaited()
