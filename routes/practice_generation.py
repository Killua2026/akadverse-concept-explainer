"""Authenticated independent practice generation endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, require_shared_secret_auth
from config import GeminiSettings
from models import GeneratePracticeRequest, GeneratePracticeResponse
from services.lesson_planning import DocumentNotFound
from services.practice_generation import generate_practice

logger = logging.getLogger(__name__)


def make_practice_generation_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/generate-practice", response_model=GeneratePracticeResponse,
        tags=["Practice"], dependencies=[Depends(require_shared_secret_auth)],
    )
    async def generate_practice_endpoint(
        body: GeneratePracticeRequest, request: Request,
    ) -> GeneratePracticeResponse:
        database = getattr(request.app.state, "ingest_database", None)
        if not settings.use_enterprise or database is None:
            raise HTTPException(503, "Practice generation is unavailable.")
        try:
            return await generate_practice(database, settings, **body.model_dump())
        except DocumentNotFound:
            raise HTTPException(404, "Document not found or has no ingested chunks") from None
        except (SQLAlchemyError, OSError):
            logger.warning("practice_generation.database_unavailable")
            raise HTTPException(503, UNAVAILABLE) from None

    return router
