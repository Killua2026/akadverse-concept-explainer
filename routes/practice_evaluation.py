from fastapi import APIRouter, Depends, HTTPException

from auth import require_shared_secret_auth
from config import GeminiSettings
from models import EvaluatePracticeRequest, EvaluatePracticeResponse
from services.practice_evaluation import evaluate_practice


def make_practice_evaluation_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/evaluate-practice",
        response_model=EvaluatePracticeResponse,
        tags=["Practice"],
        dependencies=[Depends(require_shared_secret_auth)],
    )
    async def evaluate_practice_endpoint(
        body: EvaluatePracticeRequest,
    ) -> EvaluatePracticeResponse:
        if not settings.use_enterprise and not settings.google_api_key:
            raise HTTPException(503, "Evaluation is unavailable: no LLM provider is configured.")
        return await evaluate_practice(settings, **body.model_dump())

    return router
