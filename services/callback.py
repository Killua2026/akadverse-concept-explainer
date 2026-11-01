"""Strict, bounded callback delivery to explicitly authorized hosts."""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CallbackSettings:
    callback_secret: str = field(repr=False)

    @classmethod
    def from_env(cls) -> "CallbackSettings":
        return cls(callback_secret=os.getenv("AI_SERVICE_CALLBACK_SECRET", "").strip())


async def post_callback(
    client: httpx.AsyncClient,
    *,
    url: str,
    secret: str,
    document_id: str,
    status: Literal["ingested", "failed"],
    chunk_count: int | None = None,
    error: str | None = None,
) -> bool:
    try:
        parsed = urlsplit(url)
        url_host = parsed.hostname or ""
        valid_url = parsed.scheme == "https" and bool(url_host) and not parsed.username and not parsed.password
        parsed.port
    except (TypeError, ValueError):
        url_host, valid_url = "", False
    allowed = {host.strip().lower() for host in os.getenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", "").split(",") if host.strip()}
    context = {"document_id": document_id, "url_host": url_host, "status": status}
    # Recheck authorization at delivery time, including jobs queued before validation existed.
    if not valid_url or url_host.lower() not in allowed or not secret:
        logger.warning("callback.failed_permanent", extra={**context, "attempt_number": 0, "reason": "unauthorized_destination"})
        return False
    if status == "ingested" and isinstance(chunk_count, int) and not isinstance(chunk_count, bool) and chunk_count >= 0:
        body = {"document_id": document_id, "status": status, "chunk_count": chunk_count}
    elif status == "failed" and isinstance(error, str):
        body = {"document_id": document_id, "status": status, "error": error}
    else:
        logger.warning("callback.failed_permanent", extra={**context, "attempt_number": 0, "reason": "bad_body"})
        return False
    reason = "bad_body"
    for attempt_number, delay in enumerate((0, 1, 4), start=1):
        if delay:
            await asyncio.sleep(delay)
        details = {**context, "attempt_number": attempt_number}
        try:
            response = await client.post(
                url, headers={"Authorization": f"Bearer {secret}"}, json=body,
                timeout=5.0, follow_redirects=False,
            )
        except httpx.TimeoutException:
            reason = "timeout"
        except httpx.RequestError:
            reason = "connection_error"
        else:
            details["http_status"] = response.status_code
            if response.status_code == 200:
                try:
                    data = response.json()
                except ValueError:
                    data = None
                if isinstance(data, dict) and data.get("ok") is True:
                    logger.info("callback.attempt", extra={**details, "reason": "success"})
                    return True
                reason = "bad_body"
            elif 400 <= response.status_code < 500:
                logger.warning("callback.attempt", extra={**details, "reason": "4xx"})
                logger.warning("callback.failed_permanent", extra={**details, "reason": "4xx"})
                return False
            elif 300 <= response.status_code < 400:
                reason = "redirect"
            elif response.status_code >= 500:
                reason = "5xx"
            else:
                reason = "bad_body"
        logger.warning("callback.attempt", extra={**details, "reason": reason})
    logger.warning("callback.failed_permanent", extra={**details, "reason": reason})
    return False
