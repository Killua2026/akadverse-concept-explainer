from fastapi import APIRouter, Depends, HTTPException

from auth import require_shared_secret_auth
from config import GeminiSettings
from models import GenerateHintRequest, GenerateHintResponse
from services.hint_generation import generate_hint


def make_hint_generation_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/generate-hint",
        response_model=GenerateHintResponse,
        tags=["Practice"],
        dependencies=[Depends(require_shared_secret_auth)],
    )
    async def generate_hint_endpoint(body: GenerateHintRequest) -> GenerateHintResponse:
        if not settings.use_enterprise and not settings.google_api_key:
            raise HTTPException(
                503,
                "Hint generation is unavailable: no LLM provider is configured.",
            )
        return await generate_hint(settings, **body.model_dump())

    return router
