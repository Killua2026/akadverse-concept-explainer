"""Submit signed revision batches to FSE."""

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Literal

import httpx

logger = logging.getLogger(__name__)


async def submit_questions_to_fse(
    *, webhook_url: str, webhook_secret: str, payload: dict,
) -> tuple[Literal["processed", "accepted_async", "failed_permanent", "failed_transient"], dict | None]:
    try:
        body_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        signature = hmac.new(webhook_secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
        headers = {
            "X-Akadverse-Signature": signature,
            "X-Akadverse-Timestamp": datetime.now(timezone.utc).isoformat(),
            "Idempotency-Key": f"teach_{payload['teaching_event_id']}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            for attempt, delay in enumerate((0, 1, 4), start=1):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    response = await client.post(
                        webhook_url, content=body_bytes, headers=headers, timeout=10.0,
                    )
                except httpx.RequestError as error:
                    logger.warning(
                        "submission.attempt attempt_number=%s error_type=%s",
                        attempt, type(error).__name__,
                    )
                    continue

                status = response.status_code
                logger.info("submission.attempt attempt_number=%s http_status=%s", attempt, status)
                if status in (200, 202):
                    try:
                        data = response.json()
                    except ValueError:
                        continue
                    if not isinstance(data, dict):
                        continue
                    if status == 202:
                        return "accepted_async", data
                    if data.get("status") == "processed":
                        return "processed", data
                    reason = "unexpected_body_status"
                elif status == 429 or 500 <= status < 600:
                    continue
                else:
                    reason = "4xx" if 400 <= status < 500 else "unexpected_http_status"
                logger.warning("submission.failed_permanent reason=%s", reason)
                return "failed_permanent", None
    except Exception as error:
        logger.warning("submission.failed_permanent error_type=%s", type(error).__name__)
        return "failed_permanent", None

    logger.warning("submission.failed_transient")
    return "failed_transient", None
