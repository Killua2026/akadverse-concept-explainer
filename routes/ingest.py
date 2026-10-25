"""Durable queue admission without document processing."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, require_ingest_auth
from models import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()
SUPPORTED_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
ENQUEUE_SQL = "INSERT INTO concept_explainer.jobs (document_id, payload) VALUES (:document_id, CAST(:payload AS jsonb)) ON CONFLICT (document_id) DO UPDATE SET status='queued', attempts = concept_explainer.jobs.attempts + 1, updated_at = now() RETURNING *"


@router.post("/ingest", status_code=202, response_model=IngestResponse,
             dependencies=[Depends(require_ingest_auth)])
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
