import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from config import GeminiSettings
from routes.next_action import make_next_action_router
from services import next_action

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-secret")
    settings = GeminiSettings(
        llm_provider="gemini_enterprise", google_cloud_project="test", model_names=["test-model"],
    )
    generate = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    create_client = Mock(return_value=SimpleNamespace(aio=context))
    monkeypatch.setattr(next_action, "create_client", create_client)
    app = FastAPI()
    app.include_router(make_next_action_router(settings))
    return SimpleNamespace(app=app, generate=generate, create_client=create_client)


@pytest.fixture
def model_response(configured):
    def respond(**overrides):
        decision = next_action._DecisionLLM.model_validate({
            "action": "EXPLAIN_CONCEPT",
            "target_concept_id": "ohms-law",
            "params": {},
            "reason_code": "TEST_DECISION",
            "rationale": "This action matches the latest learning event.",
            "confidence": 0.9,
            "signals": [],
            **overrides,
        })
        configured.generate.return_value = SimpleNamespace(
            parsed=decision,
            usage_metadata=SimpleNamespace(prompt_token_count=80, candidates_token_count=20),
        )

    return respond


async def post(app, token="test-secret", **body) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/next-action", headers=headers, json={
            "request_id": "request-1",
            "schema_version": "1",
            "operation": "NEXT_ACTION",
            "trigger": "CONCEPT_OPENED",
            "student_context": {},
            "current": {"concept": {"concept_id": "ohms-law"}},
            "latest_evidence": [],
            "allowed_actions": ["EXPLAIN_CONCEPT"],
            "policy": {},
            **body,
        })


@pytest.mark.parametrize(("trigger", "action"), [
    ("CONCEPT_OPENED", "EXPLAIN_CONCEPT"),
    ("ATTEMPT_EVALUATED", "REASONING_EXERCISE"),
    ("HINT_REQUESTED", "GIVE_HINT"),
    ("LESSON_COMPLETED", "COMPLETE_LESSON"),
])
async def test_next_action_returns_valid_decision(configured, model_response, trigger, action):
    model_response(action=action)

    response = await post(configured.app, trigger=trigger, allowed_actions=[action])

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["decision"]["action"] == action
    assert body["decision"]["target_concept_id"] == "ohms-law"
    assert 0 <= body["decision"]["confidence"] <= 1
    assert isinstance(body["signals"], list)
    assert body["usage"] == {
        "model": "test-model", "input_tokens": 80, "output_tokens": 20, "model_calls": 1,
    }
    prompt = configured.generate.await_args.kwargs["contents"]
    _, schema_example = prompt.rsplit("Return JSON matching this schema:\n", 1)
    assert '"params": {}' in schema_example
    assert "'params': {}" in repr(json.loads(schema_example))


async def test_next_action_respects_allowed_actions(configured, model_response, caplog):
    model_response(action="EXPLAIN_CONCEPT")

    response = await post(configured.app, allowed_actions=["FLAG_FOR_REVIEW"])

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["decision"] is None
    assert "next_action.decision_rejected reason_code=action_not_allowed" in caplog.messages


async def test_next_action_target_concept_validation(configured, model_response, caplog):
    model_response(target_concept_id="concept-99")

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["decision"] is None
    assert "next_action.decision_rejected reason_code=target_not_reachable" in caplog.messages


async def test_next_action_degraded_on_llm_failure(configured, caplog):
    configured.generate.side_effect = RuntimeError("Provider unavailable")

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "request-1",
        "schema_version": "1",
        "status": "degraded",
        "decision": None,
        "signals": [],
        "usage": {"model_calls": 0},
        "model_version": "test-model",
        "prompt_version": "next-action-v1",
    }
    assert configured.generate.await_count == 2
    assert "next_action.fallback error_type=RuntimeError reason=generation_failed" in caplog.messages


@pytest.mark.parametrize("token", [None, "wrong"])
async def test_next_action_requires_auth(configured, token):
    response = await post(configured.app, token=token)

    assert response.status_code == 401
    configured.generate.assert_not_awaited()


async def test_next_action_503_when_no_llm(configured):
    app = FastAPI()
    app.include_router(make_next_action_router(GeminiSettings(llm_provider="gemini", google_api_key="")))

    response = await post(app)

    assert response.status_code == 503
    configured.create_client.assert_not_called()
    configured.generate.assert_not_awaited()


async def test_next_action_works_without_database(configured, model_response, monkeypatch):
    configured.app.state.ingest_database = None
    monkeypatch.delenv("DATABASE_URL", raising=False)
    model_response()

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["decision"]["action"] == "EXPLAIN_CONCEPT"


async def test_next_action_signals_array(configured, model_response):
    model_response(signals=["CONCEPT_READY_FOR_REVISION"])

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json()["signals"] == ["CONCEPT_READY_FOR_REVISION"]


async def test_next_action_rejects_empty_allowed_actions(configured):
    response = await post(configured.app, allowed_actions=[])

    assert response.status_code == 422
    configured.generate.assert_not_awaited()
