from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from config import GeminiSettings
from routes.revision_questions import make_revision_questions_router
from services import revision_questions

pytestmark = pytest.mark.anyio

SHORT_ANSWER = {
    "type": "short_answer",
    "stem": "What is the SI unit of electric current?",
    "correct_text": "ampere",
    "accepted_answers": ["amp"],
    "explanation": "Electric current is measured in amperes.",
}
MCQ = {
    "type": "mcq",
    "stem": "Which quantity is measured in amperes?",
    "options": ["Voltage", "Current", "Resistance"],
    "correct_option_index": 1,
    "explanation": "Amperes measure current.",
}
TRUE_FALSE = {
    "type": "true_false",
    "stem": "Electric current is measured in volts.",
    "correct_bool": False,
    "explanation": "Current is measured in amperes.",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-inbound-key")
    monkeypatch.setenv("FSE_WEBHOOK_URL", "https://fse.example/webhooks/tutor/questions")
    monkeypatch.setenv("FSE_WEBHOOK_SECRET", "test-signing-key")
    settings = GeminiSettings(
        llm_provider="gemini_enterprise", google_cloud_project="test", model_names=["test-model"],
    )
    generate = AsyncMock()
    context = AsyncMock()
    context.__aenter__.return_value = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    monkeypatch.setattr(revision_questions, "create_client", Mock(return_value=SimpleNamespace(aio=context)))
    submit = AsyncMock(return_value=("processed", {"results": []}))
    monkeypatch.setattr(revision_questions, "submit_questions_to_fse", submit)
    app = FastAPI()
    app.include_router(make_revision_questions_router(settings))
    return SimpleNamespace(app=app, generate=generate, submit=submit)


async def post(app, token="test-inbound-key", **overrides):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        return await client.post("/generate-revision-questions", headers=headers, json={
            "teaching_event_id": "event-1",
            "student_id": "student-1",
            "concept_id": "electric-current",
            "concept_description": "Electric current and its SI unit",
            "taught_summary": "Electric current is measured in amperes.",
            "subject": "Physics",
            **overrides,
        })


async def test_generate_revision_success_short_answer(configured):
    configured.generate.return_value = SimpleNamespace(parsed={"questions": [
        SHORT_ANSWER,
        {**SHORT_ANSWER, "stem": "Name the unit used to measure electric current."},
    ]})
    results = [
        {"client_ref": "q1", "outcome": "created", "question_id": "fse-1"},
        {"client_ref": "q2", "outcome": "created", "question_id": "fse-2"},
    ]
    configured.submit.return_value = ("processed", {"results": results})

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json() == {
        "teaching_event_id": "event-1",
        "generated_count": 2,
        "webhook_attempted": True,
        "webhook_status": "processed",
        "webhook_results": results,
        "needs_review": False,
        "error_detail": None,
    }
    payload = configured.submit.await_args.kwargs["payload"]
    assert payload["student_id"] == "student-1"
    assert payload["context"]["concept_title"] == "Electric current and its SI unit"
    assert payload["questions"][0]["correct_answer"] == "ampere"
    assert payload["questions"][0]["accepted_answers"] == ["amp"]


async def test_generate_revision_success_mcq(configured):
    configured.generate.return_value = SimpleNamespace(parsed={"questions": [MCQ]})

    response = await post(configured.app, question_types=["mcq"])

    assert response.status_code == 200
    assert response.json()["webhook_status"] == "processed"
    assert response.json()["needs_review"] is True
    question, = configured.submit.await_args.kwargs["payload"]["questions"]
    assert question["options"] == [
        {"key": "A", "text": "Voltage"},
        {"key": "B", "text": "Current"},
        {"key": "C", "text": "Resistance"},
    ]
    assert question["correct_answer"] == "B"


async def test_generate_revision_success_true_false(configured):
    configured.generate.return_value = SimpleNamespace(parsed={"questions": [TRUE_FALSE]})

    response = await post(configured.app, question_types=["true_false"])

    assert response.status_code == 200
    question, = configured.submit.await_args.kwargs["payload"]["questions"]
    assert question["correct_answer"] is False


async def test_generate_revision_mixed_types(configured):
    configured.generate.return_value = SimpleNamespace(parsed={"questions": [MCQ, SHORT_ANSWER, TRUE_FALSE]})
    types = ["mcq", "short_answer", "true_false"]

    response = await post(configured.app, question_types=types)

    assert response.status_code == 200
    questions = configured.submit.await_args.kwargs["payload"]["questions"]
    assert [question["type"] for question in questions] == types
    assert [question["client_ref"] for question in questions] == ["q1", "q2", "q3"]


async def test_generate_revision_filters_untyped(configured):
    configured.generate.return_value = SimpleNamespace(parsed={"questions": [MCQ, SHORT_ANSWER]})

    response = await post(configured.app, question_types=["short_answer"])

    assert response.status_code == 200
    question, = configured.submit.await_args.kwargs["payload"]["questions"]
    assert question["type"] == "short_answer"
    assert question["client_ref"] == "q1"


async def test_generate_revision_webhook_failure_returns_degraded(configured):
    configured.generate.return_value = SimpleNamespace(parsed={"questions": [
        SHORT_ANSWER,
        {**SHORT_ANSWER, "stem": "Name the unit used to measure electric current."},
    ]})
    configured.submit.return_value = ("failed_transient", None)

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json() == {
        "teaching_event_id": "event-1",
        "generated_count": 2,
        "webhook_attempted": True,
        "webhook_status": "failed_transient",
        "webhook_results": [],
        "needs_review": True,
        "error_detail": "webhook failed after 3 attempts",
    }


async def test_generate_revision_llm_failure_returns_degraded(configured):
    configured.generate.side_effect = RuntimeError("provider failed")

    response = await post(configured.app)

    assert response.status_code == 200
    assert response.json() == {
        "teaching_event_id": "event-1",
        "generated_count": 0,
        "webhook_attempted": False,
        "webhook_status": "not_attempted",
        "webhook_results": [],
        "needs_review": True,
        "error_detail": "generation failed",
    }
    configured.submit.assert_not_awaited()


async def test_generate_revision_missing_config_returns_503(configured, monkeypatch):
    monkeypatch.delenv("FSE_WEBHOOK_URL")

    response = await post(configured.app)

    assert response.status_code == 503
    assert response.json() == {"detail": "Revision submission is not configured."}
    configured.generate.assert_not_awaited()


@pytest.mark.parametrize("token", [None, "wrong"])
async def test_generate_revision_requires_auth(configured, token):
    response = await post(configured.app, token)

    assert response.status_code == 401
    configured.generate.assert_not_awaited()
    configured.submit.assert_not_awaited()
