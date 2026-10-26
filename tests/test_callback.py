"""Callback destination admission and error disclosure checks."""

import asyncio
import json
import logging
import socket
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from routes import ingest
from workers.ingestion import sanitize_error
from services.callback import post_callback


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("url,detail", [
    ("http://example.com/callback", "callback_url must use https"),
    ("not a URL", "callback_url must be a valid https URL"),
    ("https://", "callback_url must be a valid https URL"),
    ("https://example.com:bad", "callback_url must be a valid https URL"),
    ("https://user:password@example.com", "callback_url must be a valid https URL"),
    (42, "callback_url must be a valid https URL"),
])
async def test_invalid_callback_rejected_before_queueing(monkeypatch, url, detail):
    monkeypatch.setenv("DATABASE_URL", "configured")
    monkeypatch.setenv("AI_SERVICE_API_KEY", "test-inbound")
    app = FastAPI()
    app.include_router(ingest.router)
    app.state.ingest_status = "ready"
    sessions = Mock()
    app.state.ingest_database = SimpleNamespace(sessions=sessions)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ingest", json={"callback_url": url},
                                     headers={"Authorization": "Bearer test-inbound"})
    assert response.status_code == 422
    assert response.json() == {"detail": detail}
    sessions.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("address", [
    "10.0.0.1", "172.16.0.1", "192.168.1.1", "127.0.0.2", "169.254.169.254",
    "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1",
])
async def test_disallowed_callback_address(monkeypatch, address):
    monkeypatch.setattr(ingest, "resolve_callback_host", AsyncMock(return_value=[
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
    ]))
    with pytest.raises(HTTPException) as captured:
        await ingest.validate_callback_url("https://example.com/callback")
    assert captured.value.status_code == 422
    assert captured.value.detail == "callback_url resolves to a disallowed address"


@pytest.mark.anyio
async def test_callback_allowlist(monkeypatch):
    monkeypatch.setenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", " EXAMPLE.COM, other.example ")
    await ingest.validate_callback_url("https://example.com/callback")
    with pytest.raises(HTTPException) as captured:
        await ingest.validate_callback_url("https://unapproved.example/callback")
    assert captured.value.detail == "callback_url host not allowed"


@pytest.mark.anyio
async def test_callback_mixed_dns_answers_rejected(monkeypatch):
    monkeypatch.setattr(ingest, "resolve_callback_host", AsyncMock(return_value=[
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
    ]))
    with pytest.raises(HTTPException) as captured:
        await ingest.validate_callback_url("https://example.com/callback")
    assert captured.value.detail == "callback_url resolves to a disallowed address"


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [TimeoutError(), socket.gaierror()])
async def test_callback_dns_failure(monkeypatch, failure):
    monkeypatch.setattr(ingest, "resolve_callback_host", AsyncMock(side_effect=failure))
    with pytest.raises(HTTPException) as captured:
        await ingest.validate_callback_url("https://example.com/callback")
    assert captured.value.detail == "callback_url resolves to a disallowed address"


@pytest.mark.anyio
async def test_callback_dns_timeout_budget(monkeypatch):
    original = asyncio.wait_for
    timeouts = []

    async def bounded(awaitable, timeout):
        timeouts.append(timeout)
        return await original(awaitable, timeout=timeout)

    monkeypatch.setattr(ingest.asyncio, "wait_for", bounded)
    await ingest.validate_callback_url("https://example.com/callback")
    assert timeouts == [2.0]


@pytest.mark.parametrize("message", [
    r"Failed at C:\private\document", "Failed /home/user/file", "Failed /tmp/file", "Failed /var/data/file",
    " ".join(["a" * 21] * 4),
])
def test_unsafe_error_replaced(message):
    assert sanitize_error(message) == "Processing failed (see server logs)"


def test_error_normalized_and_capped():
    assert sanitize_error(" Extraction\n failed\t safely ") == "Extraction failed safely"
    assert sanitize_error("x" * 600) == "x" * 500


async def deliver(
    monkeypatch, outcomes, *, status: Literal["ingested", "failed"] = "ingested",
    chunk_count: int | None = 2, error: str | None = None,
):
    requests = []
    sleep = AsyncMock()
    monkeypatch.setattr("services.callback.asyncio.sleep", sleep)

    def handle(request):
        requests.append(request)
        outcome = outcomes[min(len(requests) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    # An externally configured client must still have redirects disabled per request.
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
        result = await post_callback(
            client, url="https://example.com/callback?private=value", secret="callback-test-secret",
            document_id="doc-1", status=status, chunk_count=chunk_count, error=error,
        )
    return result, requests, sleep


@pytest.mark.anyio
async def test_success_on_ok_true(monkeypatch):
    result, requests, sleep = await deliver(monkeypatch, [httpx.Response(200, json={"ok": True})])
    assert result is True and len(requests) == 1
    sleep.assert_not_awaited()


@pytest.mark.anyio
async def test_success_on_ok_true_with_ignored(monkeypatch):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(200, json={"ok": True, "ignored": "unknown document_id"})])
    assert result is True and len(requests) == 1


@pytest.mark.anyio
async def test_retry_on_500_then_success(monkeypatch):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(500), httpx.Response(200, json={"ok": True})])
    assert result is True and len(requests) == 2


@pytest.mark.anyio
async def test_no_retry_on_400(monkeypatch):
    result, requests, sleep = await deliver(monkeypatch, [httpx.Response(400)])
    assert result is False and len(requests) == 1
    sleep.assert_not_awaited()


@pytest.mark.anyio
async def test_no_retry_on_404(monkeypatch):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(404)])
    assert result is False and len(requests) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("code", [401, 403])
async def test_other_4xx_not_retried(monkeypatch, code):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(code)])
    assert result is False and len(requests) == 1


@pytest.mark.anyio
async def test_retry_on_timeout(monkeypatch):
    result, requests, _ = await deliver(monkeypatch, [httpx.ReadTimeout("simulated"), httpx.Response(200, json={"ok": True})])
    assert result is True and len(requests) == 2


@pytest.mark.anyio
async def test_retry_on_connection_error(monkeypatch):
    result, requests, _ = await deliver(monkeypatch, [httpx.ConnectError("simulated"), httpx.Response(200, json={"ok": True})])
    assert result is True and len(requests) == 2


@pytest.mark.anyio
async def test_redirect_not_followed(monkeypatch):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(302, headers={"Location": "https://unlisted.example/callback"})])
    assert result is False and len(requests) == 3
    assert all(request.url.host == "example.com" for request in requests)


@pytest.mark.anyio
async def test_200_without_ok_field_retried(monkeypatch):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(200, json={"status": "received"})])
    assert result is False and len(requests) == 3


@pytest.mark.anyio
async def test_200_with_malformed_json_retried(monkeypatch):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(200, text="not json")])
    assert result is False and len(requests) == 3


@pytest.mark.anyio
@pytest.mark.parametrize("body", [{"ok": False}, {"ok": 1}, {"ok": "true"}, [], None])
async def test_ok_must_be_json_boolean_true(monkeypatch, body):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(200, json=body)])
    assert result is False and len(requests) == 3


@pytest.mark.anyio
@pytest.mark.parametrize("code", [201, 202, 204])
async def test_other_2xx_retried(monkeypatch, code):
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(code)])
    assert result is False and len(requests) == 3


@pytest.mark.anyio
async def test_exhaustion_returns_false(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="services.callback")
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(500)])
    assert result is False and len(requests) == 3
    records = [record for record in caplog.records if record.name == "services.callback"]
    assert records[-1].message == "callback.failed_permanent"
    assert records[-1].document_id == "doc-1" and records[-1].reason == "5xx"
    for record in records:
        assert record.url_host == "example.com"
        assert record.status == "ingested"
        assert record.attempt_number in {1, 2, 3}
        assert record.http_status == 500
        assert "callback-test-secret" not in repr(record.__dict__)
        assert "private=value" not in repr(record.__dict__)


@pytest.mark.anyio
async def test_auth_header_sent(monkeypatch):
    _, requests, _ = await deliver(monkeypatch, [httpx.Response(200, json={"ok": True})])
    assert requests[0].headers.get_list("authorization") == ["Bearer callback-test-secret"]
    assert "proxy-authorization" not in requests[0].headers
    assert requests[0].extensions["timeout"] == {name: 5.0 for name in ("connect", "read", "write", "pool")}


@pytest.mark.anyio
async def test_ingested_body(monkeypatch):
    _, requests, _ = await deliver(monkeypatch, [httpx.Response(200, json={"ok": True})], error="must be omitted")
    assert json.loads(requests[0].content) == {"document_id": "doc-1", "status": "ingested", "chunk_count": 2}


@pytest.mark.anyio
async def test_failed_body(monkeypatch):
    _, requests, _ = await deliver(monkeypatch, [httpx.Response(200, json={"ok": True})], status="failed", error="Safe error")
    assert json.loads(requests[0].content) == {"document_id": "doc-1", "status": "failed", "error": "Safe error"}


@pytest.mark.anyio
async def test_backoff_delays(monkeypatch):
    _, _, sleep = await deliver(monkeypatch, [httpx.Response(500)])
    assert [args.args[0] for args in sleep.await_args_list] == [1, 4]


@pytest.mark.anyio
@pytest.mark.parametrize("allowed", ["", "unlisted.example", "*.example.com"])
async def test_delivery_requires_explicit_authorization(monkeypatch, allowed):
    monkeypatch.setenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", allowed)
    result, requests, _ = await deliver(monkeypatch, [httpx.Response(200, json={"ok": True})])
    assert result is False and requests == []
