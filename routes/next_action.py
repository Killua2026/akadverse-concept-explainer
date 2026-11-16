from fastapi import APIRouter, Depends, HTTPException

from auth import require_shared_secret_auth
from config import GeminiSettings
from models import NextActionRequest, NextActionResponse
from services.next_action import decide_next_action


def make_next_action_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/next-action",
        response_model=NextActionResponse,
        tags=["Orchestrator"],
        dependencies=[Depends(require_shared_secret_auth)],
    )
    async def next_action_endpoint(body: NextActionRequest) -> NextActionResponse:
        if not settings.use_enterprise and not settings.google_api_key:
            raise HTTPException(
                503,
                "Next action decisions are unavailable: no LLM provider is configured.",
            )
        return await decide_next_action(settings, **body.model_dump())

    return router
