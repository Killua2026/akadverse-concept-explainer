"""Synchronous removal of document chunks and ingestion jobs."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, require_ingest_auth
from models import DocumentStatusResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.delete("/documents/{document_id}", status_code=204,
               dependencies=[Depends(require_ingest_auth)])
async def delete_document(document_id: str, request: Request) -> Response:
    try:
        async with request.app.state.ingest_database.sessions() as session, session.begin():
            chunks = await session.execute(text(
                "DELETE FROM concept_explainer.chunks WHERE document_id = :document_id"
            ), {"document_id": document_id})
            jobs = await session.execute(text(
                "DELETE FROM concept_explainer.jobs WHERE document_id = :document_id"
            ), {"document_id": document_id})
            deleted = chunks.rowcount > 0 or jobs.rowcount > 0
    except (SQLAlchemyError, OSError):
        logger.warning("document.delete_failed")
        raise HTTPException(503, UNAVAILABLE) from None
    if not deleted:
        raise HTTPException(404, "Document not found")
    return Response(status_code=204)


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse,
            dependencies=[Depends(require_ingest_auth)])
async def document_status(document_id: str, request: Request) -> DocumentStatusResponse:
    try:
        async with request.app.state.ingest_database.sessions() as session:
            result = await session.execute(text(
                "SELECT id, document_id, status, attempts, last_error, created_at, updated_at "
                "FROM concept_explainer.jobs WHERE document_id = :document_id"
            ), {"document_id": document_id})
            job = result.mappings().first()
            if job is None:
                raise HTTPException(404, "Document not found")
            chunk_count = await session.scalar(text(
                "SELECT count(*) FROM concept_explainer.chunks WHERE document_id = :document_id"
            ), {"document_id": document_id})
    except (SQLAlchemyError, OSError):
        logger.warning("document.status_failed")
        raise HTTPException(503, UNAVAILABLE) from None
    return DocumentStatusResponse(**{key: value for key, value in job.items() if key != "id"},
                                  chunk_count=chunk_count)
