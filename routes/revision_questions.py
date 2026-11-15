import os

from fastapi import APIRouter, Depends, HTTPException

from auth import require_shared_secret_auth
from config import GeminiSettings
from models import GenerateRevisionQuestionsRequest, GenerateRevisionQuestionsResponse
from services.revision_questions import generate_revision_questions


def make_revision_questions_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/generate-revision-questions",
        response_model=GenerateRevisionQuestionsResponse,
        tags=["Revision"],
        dependencies=[Depends(require_shared_secret_auth)],
    )
    async def generate_revision_questions_endpoint(
        body: GenerateRevisionQuestionsRequest,
    ) -> GenerateRevisionQuestionsResponse:
        webhook_url = os.getenv("FSE_WEBHOOK_URL", "").strip()
        webhook_secret = os.getenv("FSE_WEBHOOK_SECRET", "").strip()
        if not webhook_url or not webhook_secret:
            raise HTTPException(503, "Revision submission is not configured.")
        if not settings.use_enterprise and not settings.google_api_key:
            raise HTTPException(503, "Revision generation is unavailable: no LLM provider configured.")
        return await generate_revision_questions(
            settings, webhook_url=webhook_url, webhook_secret=webhook_secret, **body.model_dump(),
        )

    return router
