"""Synchronous removal of document chunks and ingestion jobs."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, require_ingest_auth

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
