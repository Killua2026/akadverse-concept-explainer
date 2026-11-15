from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from config import GeminiSettings
from routes.hint_generation import make_hint_generation_router
from services import hint_generation

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
    monkeypatch.setattr(hint_generation, "create_client", Mock(return_value=SimpleNamespace(aio=context)))
    app = FastAPI()
    app.include_router(make_hint_generation_router(settings))
    return SimpleNamespace(app=app, generate=generate)


@pytest.fixture
def valid_result():
    return SimpleNamespace(parsed=hint_generation._HintResult(
        hint_content="Consider how the quantities relate.",
        rationale="The hint focuses on the relationship between quantities.",
    ))


async def post(app, token="test-secret", **body) -> httpx.Response:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/generate-hint", headers=headers, json={
            "concept_id": "ohms-law",
            "concept_description": "Apply Ohm's law.",
            "question": "What is the current in amps given 12 volts and 4 ohms?",
            "expected_answer": "3",
            "evaluation_criteria": ["Divide voltage by resistance."],
            "student_attempt": "I multiplied 12 by 4.",
            "target_level": 1,
            "previous_hints": [],
            **body,
        })


@pytest.mark.parametrize(("target_level", "hint_type", "phrase"), [
    (1, "L1_METACOGNITIVE", "Ask ONE question that helps the student inspect their approach."),
    (2, "L2_DIRECTIONAL", "WITHOUT giving the answer or revealing a specific formula."),
    (3, "L3_CONCEPT_REMINDER", "do NOT apply it to the student's specific numbers yet."),
    (4, "L4_ANALOGOUS_EXAMPLE", "STRUCTURALLY SIMILAR but DIFFERENT"),
    (5, "L5_PARTIAL_SOLUTION", "Work through the FIRST part"),
    (6, "L6_FULL_SOLUTION", "Reveal the answer and explain why each step is correct."),
])
async def test_generate_hint_each_level(configured, valid_result, target_level, hint_type, phrase):
    configured.generate.return_value = valid_result

    response = await post(configured.app, target_level=target_level)

    assert response.status_code == 200
    body = response.json()
    assert body["level_given"] == target_level
    assert body["hint_type"] == hint_type
    assert body["hint_content"]
    assert body["rationale"]
    assert body["needs_review"] is False
    call = configured.generate.await_args.kwargs
    assert phrase in call["contents"]
    assert call["config"].temperature == 0.3
    assert call["config"].response_schema is not None
    assert call["config"].response_mime_type == "application/json"


@pytest.mark.parametrize("target_level", [0, 7])
async def test_generate_hint_invalid_level_returns_422(configured, target_level):
    response = await post(configured.app, target_level=target_level)

    assert response.status_code == 422
    configured.generate.assert_not_awaited()


async def test_generate_hint_llm_failure_returns_degraded(configured):
    configured.generate.side_effect = RuntimeError("Provider unavailable")

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json() == {
        "level_given": -1,
        "hint_type": "NO_HINT",
        "hint_content": "",
        "rationale": "Hint generation is temporarily unavailable.",
        "needs_review": True,
    }
    assert configured.generate.await_count == 2


@pytest.mark.parametrize("token", [None, "wrong"])
async def test_generate_hint_requires_auth(configured, token):
    response = await post(configured.app, token=token)

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    configured.generate.assert_not_awaited()


async def test_generate_hint_works_without_database(configured, monkeypatch, valid_result):
    configured.app.state.ingest_database = None
    monkeypatch.delenv("DATABASE_URL", raising=False)
    configured.generate.return_value = valid_result

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json() == {
        "level_given": 1,
        "hint_type": "L1_METACOGNITIVE",
        "hint_content": valid_result.parsed.hint_content,
        "rationale": valid_result.parsed.rationale,
        "needs_review": False,
    }
