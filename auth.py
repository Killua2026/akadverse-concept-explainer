"""Shared-secret authentication and ingestion readiness checks."""

import os
import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

UNCONFIGURED = "Ingestion is not configured on this service. Contact the AI team."
UNAVAILABLE = "Ingestion is temporarily unavailable. Retry with backoff."
shared_secret_bearer = HTTPBearer(
    scheme_name="SharedSecretBearer",
    description="Enter your AI_SERVICE_API_KEY value without the Bearer prefix.",
    auto_error=False,
)


async def require_ingest_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(shared_secret_bearer),
) -> None:
    token = os.getenv("AI_SERVICE_API_KEY", "").strip()
    if not os.getenv("DATABASE_URL", "").strip() or not token:
        raise HTTPException(503, UNCONFIGURED)
    state = getattr(request.app.state, "ingest_status", "disabled (db unreachable)")
    if state != "ready":
        detail = UNCONFIGURED if state.startswith("disabled (missing ") else UNAVAILABLE
        raise HTTPException(503, detail)
    await require_shared_secret_auth(request, credentials)


async def require_shared_secret_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(shared_secret_bearer),
) -> None:
    """Authenticate callers independently of ingestion configuration or state."""
    token = os.getenv("AI_SERVICE_API_KEY", "").strip()
    if not token:
        raise HTTPException(503, "Shared-secret authentication is not configured on this service.")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), token.encode()
    ):
        raise HTTPException(401, "Unauthorized")
