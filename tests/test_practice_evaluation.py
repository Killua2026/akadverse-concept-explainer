from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from config import GeminiSettings
from routes.practice_evaluation import make_practice_evaluation_router
from services import practice_evaluation

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
    monkeypatch.setattr(practice_evaluation, "create_client", Mock(return_value=SimpleNamespace(aio=context)))
    app = FastAPI()
    app.include_router(make_practice_evaluation_router(settings))
    return SimpleNamespace(app=app, generate=generate)


@pytest.fixture
def strong_result():
    return {
        "answer_correct": True,
        "answer_feedback": "You calculated the current correctly.",
        "reasoning_quality": "strong",
        "reasoning_feedback": "You correctly applied Ohm's law by dividing voltage by resistance.",
        "misconceptions": [],
        "evidence_strength": "independent",
        "needs_review": False,
        "recommended_next_step": "progress",
    }


async def post(app, token="test-secret", **body):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/evaluate-practice", headers=headers, json={
            "concept_id": "ohms-law",
            "concept_description": "Apply Ohm's law.",
            "question": "What is the current in amps given 12 volts and 4 ohms?",
            "expected_answer": "3",
            "evaluation_criteria": ["Divide voltage by resistance."],
            "student_answer": "3.0",
            "student_reasoning": "Ohm's law gives I = V/R, so I divided 12 by 4.",
            "response_type": "number",
            **body,
        })


async def test_evaluate_practice_correct_answer(configured, strong_result):
    configured.generate.return_value = SimpleNamespace(
        parsed=practice_evaluation._EvaluationResult(**strong_result),
    )

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json() == {"concept_id": "ohms-law", **strong_result}
    call = configured.generate.await_args.kwargs
    assert call["model"] == "test-model"
    assert call["config"].temperature == 0.1
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is practice_evaluation._EvaluationResult


async def test_evaluate_practice_incorrect_answer(configured, strong_result):
    misconception = {"label": "Multiplies voltage by resistance", "evidence": "I multiplied 12 by 4."}
    configured.generate.return_value = SimpleNamespace(parsed={
        **strong_result,
        "answer_correct": False,
        "answer_feedback": "Check which operation Ohm's law requires.",
        "reasoning_quality": "weak",
        "reasoning_feedback": "Divide voltage by resistance to calculate current.",
        "misconceptions": [misconception],
        "recommended_next_step": "reinforce",
    })

    response = await post(configured.app, student_answer="48", student_reasoning=misconception["evidence"])

    assert response.status_code == 200
    body = response.json()
    assert body["answer_correct"] is False
    assert body["reasoning_quality"] == "weak"
    assert body["recommended_next_step"] == "reinforce"
    assert body["misconceptions"] == [misconception]


async def test_evaluate_practice_borderline_needs_review(configured, strong_result):
    configured.generate.return_value = SimpleNamespace(parsed={
        **strong_result,
        "answer_correct": False,
        "answer_feedback": "Give one final current value.",
        "reasoning_quality": "weak",
        "reasoning_feedback": "Show which operation you used.",
        "needs_review": True,
        "evidence_strength": "insufficient",
        "recommended_next_step": "reinforce",
    })

    response = await post(configured.app, student_answer="3 or 4", student_reasoning="I guessed.")

    assert response.status_code == 200
    assert response.json()["needs_review"] is True
    assert response.json()["evidence_strength"] == "insufficient"


def degraded_body(concept_id):
    return {
        "concept_id": concept_id,
        "answer_correct": False,
        "answer_feedback": (
            "Automatic evaluation is temporarily unavailable. "
            "Please review this answer with your instructor or try again."
        ),
        "reasoning_quality": "absent",
        "reasoning_feedback": "Automatic reasoning evaluation is temporarily unavailable.",
        "misconceptions": [],
        "evidence_strength": "insufficient",
        "needs_review": True,
        "recommended_next_step": "reinforce",
    }


async def test_evaluate_practice_provider_failure_returns_degraded(configured):
    configured.generate.side_effect = RuntimeError("Provider unavailable")
    concept_id = "provider-failure-concept"

    response = await post(configured.app, concept_id=concept_id)

    assert response.status_code == 200
    assert response.json() == degraded_body(concept_id)
    assert configured.generate.await_count == 2


async def test_evaluate_practice_parsed_read_failure_returns_degraded(configured, caplog):
    class UnreadableResponse(SimpleNamespace):
        @property
        def parsed(self):
            raise AttributeError("Parsed response is unavailable")

    configured.generate.side_effect = [
        RuntimeError("Structured generation unavailable"),
        UnreadableResponse(text="This response contains prose instead of JSON."),
    ]
    concept_id = "parsed-read-failure-concept"

    response = await post(configured.app, concept_id=concept_id)

    assert response.status_code == 200
    assert response.json() == degraded_body(concept_id)
    assert configured.generate.await_count == 2
    assert "evaluate_practice.response_read_failed error_type=AttributeError" in caplog.messages


@pytest.mark.parametrize("token", [None, "wrong"])
async def test_evaluate_practice_requires_auth(configured, token):
    response = await post(configured.app, token=token)

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    configured.generate.assert_not_awaited()


async def test_evaluate_works_without_database(configured, monkeypatch, strong_result):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    configured.app.state.ingest_database = None
    configured.generate.return_value = SimpleNamespace(
        parsed=practice_evaluation._EvaluationResult(**strong_result),
    )

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json() == {"concept_id": "ohms-law", **strong_result}
