"""Authenticated lesson planning endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, require_shared_secret_auth
from config import GeminiSettings
from models import AnalyseMaterialRequest, AnalyseMaterialResponse
from services.lesson_planning import DocumentNotFound, analyse_material

logger = logging.getLogger(__name__)


def make_lesson_planning_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/analyse-material", response_model=AnalyseMaterialResponse, tags=["Lesson Planning"],
        dependencies=[Depends(require_shared_secret_auth)],
    )
    async def analyse_material_endpoint(body: AnalyseMaterialRequest, request: Request) -> AnalyseMaterialResponse:
        database = getattr(request.app.state, "ingest_database", None)
        if not settings.use_enterprise or database is None:
            raise HTTPException(503, "Course material retrieval is unavailable.")
        try:
            return await analyse_material(
                database, settings, document_id=body.document_id, max_concepts=body.max_concepts,
            )
        except DocumentNotFound:
            raise HTTPException(404, "Document not found or has no ingested chunks") from None
        except (SQLAlchemyError, OSError):
            logger.warning("lesson_planning.database_unavailable")
            raise HTTPException(503, UNAVAILABLE) from None

    return router
