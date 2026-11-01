"""Durable queue admission without document processing."""

import logging
import asyncio
import ipaddress
import os
import socket
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, UNCONFIGURED, require_ingest_auth
from models import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()
SUPPORTED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
ENQUEUE_SQL = "INSERT INTO concept_explainer.jobs (document_id, payload) VALUES (:document_id, CAST(:payload AS jsonb)) ON CONFLICT (document_id) DO UPDATE SET status='queued', attempts = concept_explainer.jobs.attempts + 1, updated_at = now() RETURNING *"


async def resolve_callback_host(host: str, port: int):
    return await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)


async def validate_callback_url(url: str) -> None:
    """Validate callback destinations before durable queue admission."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
        if not hostname or parsed.scheme not in {"https", "http"} or parsed.username or parsed.password:
            raise ValueError("Invalid callback URL")
        if any(character.isspace() for character in url) or "\\" in url:
            raise ValueError("Invalid callback URL")
    except (ValueError, TypeError):
        raise HTTPException(422, "callback_url must be a valid https URL") from None
    if not url.startswith("https://"):
        raise HTTPException(422, "callback_url must use https")
    allowed = {
        host.strip().lower() for host in os.getenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", "").split(",")
        if host.strip()
    }
    if not allowed:
        raise HTTPException(503, UNCONFIGURED)
    if hostname.lower() not in allowed:
        raise HTTPException(422, "callback_url host not allowed")
    try:
        addresses = await asyncio.wait_for(
            resolve_callback_host(hostname, port or 443),
            timeout=2.0,
        )
        # Only the explicitly authorized local development names bypass the
        # private-address guard. Other listed names retain the DNS typo guard.
        local_development = hostname.lower() in {"localhost", "127.0.0.1"}
        if not addresses or (not local_development and any(
            not ipaddress.ip_address(address[4][0]).is_global for address in addresses
        )):
            raise HTTPException(422, "callback_url resolves to a disallowed address")
    except (OSError, TimeoutError, ValueError):
        raise HTTPException(422, "callback_url resolves to a disallowed address") from None


async def validate_callback_request(request: Request) -> None:
    if not any(host.strip() for host in os.getenv("AI_SERVICE_CALLBACK_ALLOWED_HOSTS", "").split(",")):
        request.app.state.ingest_status = "disabled (missing AI_SERVICE_CALLBACK_ALLOWED_HOSTS)"
        raise HTTPException(503, UNCONFIGURED)
    # Inspect the raw value so malformed URLs receive the callback contract error,
    # rather than Pydantic's generic HttpUrl validation response.
    try:
        body = await request.json()
    except ValueError:
        return
    if isinstance(body, dict) and "callback_url" in body:
        if not isinstance(body["callback_url"], str):
            raise HTTPException(422, "callback_url must be a valid https URL")
        await validate_callback_url(body["callback_url"])


@router.post("/ingest", status_code=202, response_model=IngestResponse,
             dependencies=[Depends(require_ingest_auth), Depends(validate_callback_request)])
async def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    if body.mime_type not in SUPPORTED_TYPES:
        raise HTTPException(422, "Unsupported file type. Supported: PDF, DOCX, PPTX.")
    try:
        async with request.app.state.ingest_database.sessions() as session:
            await session.execute(text(ENQUEUE_SQL), {
                "document_id": body.document_id, "payload": body.model_dump_json(),
            })
            await session.commit()
    except (SQLAlchemyError, OSError):
        # Driver exception text and parameters may contain credentials.
        logger.warning("Ingestion enqueue failed: database unavailable.")
        request.app.state.ingest_status = "disabled (db unreachable)"
        raise HTTPException(503, UNAVAILABLE) from None
    return IngestResponse(status="queued", document_id=body.document_id)
