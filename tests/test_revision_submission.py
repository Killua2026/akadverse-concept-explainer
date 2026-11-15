import hashlib
import hmac
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest

from services import revision_submission

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def setup_submission(monkeypatch):
    async_client = httpx.AsyncClient
    sleep = AsyncMock()
    monkeypatch.setattr(revision_submission.asyncio, "sleep", sleep)

    def setup(*responses):
        handler = AsyncMock(side_effect=responses)
        monkeypatch.setattr(
            revision_submission.httpx,
            "AsyncClient",
            lambda **kwargs: async_client(transport=httpx.MockTransport(handler), **kwargs),
        )
        return handler

    return setup


async def submit():
    return await revision_submission.submit_questions_to_fse(
        webhook_url="https://fse.example/webhooks/tutor/questions",
        webhook_secret="test-signing-key",
        payload={"teaching_event_id": "event-1", "questions": [{"stem": "A question?"}]},
    )


async def test_hmac_signature_matches_expected(setup_submission):
    handler = setup_submission(httpx.Response(200, json={"status": "processed"}))

    await submit()

    request = handler.await_args.args[0]
    expected_body = b'{"questions":[{"stem":"A question?"}],"teaching_event_id":"event-1"}'
    assert request.content == expected_body
    assert request.headers["X-Akadverse-Signature"] == hmac.new(
        b"test-signing-key", expected_body, hashlib.sha256,
    ).hexdigest()
    assert datetime.fromisoformat(request.headers["X-Akadverse-Timestamp"]).utcoffset() == timedelta(0)
    assert request.headers["Content-Type"] == "application/json"
    assert "Authorization" not in request.headers


async def test_idempotency_key_format(setup_submission):
    handler = setup_submission(httpx.Response(200, json={"status": "processed"}))

    await submit()

    assert handler.await_args.args[0].headers["Idempotency-Key"] == "teach_event-1"


async def test_200_returns_processed(setup_submission):
    body = {"status": "processed", "results": [{"outcome": "created", "question_id": "fse-1"}]}
    setup_submission(httpx.Response(200, json=body))

    assert await submit() == ("processed", body)


async def test_4xx_does_not_retry(setup_submission):
    handler = setup_submission(httpx.Response(400))

    assert await submit() == ("failed_permanent", None)
    assert handler.await_count == 1


async def test_5xx_retries_3_times_then_fails(setup_submission):
    handler = setup_submission(*(httpx.Response(500) for _ in range(3)))

    assert await submit() == ("failed_transient", None)
    assert handler.await_count == 3
    first, *retries = [call.args[0] for call in handler.await_args_list]
    for retry in retries:
        assert retry.content == first.content
        assert retry.headers == first.headers
    assert [call.args for call in revision_submission.asyncio.sleep.await_args_list] == [(1,), (4,)]


async def test_429_retries(setup_submission):
    body = {"status": "processed", "results": []}
    handler = setup_submission(httpx.Response(429), httpx.Response(200, json=body))

    assert await submit() == ("processed", body)
    assert handler.await_count == 2


async def test_200_malformed_body_retries(setup_submission):
    handler = setup_submission(*(httpx.Response(200, text="not json") for _ in range(3)))

    assert await submit() == ("failed_transient", None)
    assert handler.await_count == 3
