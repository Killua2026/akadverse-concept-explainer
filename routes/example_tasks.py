"""Authenticated worked-example task endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, require_shared_secret_auth
from config import GeminiSettings
from models import GenerateExampleTasksRequest, GenerateExampleTasksResponse
from services.example_tasks import generate_example_tasks
from services.lesson_planning import DocumentNotFound

logger = logging.getLogger(__name__)


def make_example_tasks_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/generate-example-tasks", response_model=GenerateExampleTasksResponse,
        tags=["Lesson Planning"], dependencies=[Depends(require_shared_secret_auth)],
    )
    async def generate_example_tasks_endpoint(
        body: GenerateExampleTasksRequest, request: Request,
    ) -> GenerateExampleTasksResponse:
        database = getattr(request.app.state, "ingest_database", None)
        if not settings.use_enterprise or database is None:
            raise HTTPException(503, "Course material retrieval is unavailable.")
        try:
            return await generate_example_tasks(database, settings, **body.model_dump())
        except DocumentNotFound:
            raise HTTPException(404, "Document not found or has no ingested chunks") from None
        except (SQLAlchemyError, OSError):
            logger.warning("example_tasks.database_unavailable")
            raise HTTPException(503, UNAVAILABLE) from None

    return router
