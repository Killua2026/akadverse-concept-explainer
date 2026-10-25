"""Shared-secret authentication for ingestion."""

import os
import secrets

from fastapi import HTTPException, Request

UNCONFIGURED = "Ingestion is not configured on this service. Contact the AI team."
UNAVAILABLE = "Ingestion is temporarily unavailable. Retry with backoff."


async def require_ingest_auth(request: Request) -> None:
    token = os.getenv("AI_SERVICE_API_KEY", "").strip()
    if not os.getenv("DATABASE_URL", "").strip() or not token:
        raise HTTPException(503, UNCONFIGURED)
    state = getattr(request.app.state, "ingest_status", "disabled (db unreachable)")
    if state != "ready":
        detail = UNCONFIGURED if state.startswith("disabled (missing ") else UNAVAILABLE
        raise HTTPException(503, detail)
    scheme, _, supplied = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not supplied or not secrets.compare_digest(
        supplied.encode(), token.encode()
    ):
        raise HTTPException(401, "Unauthorized")
