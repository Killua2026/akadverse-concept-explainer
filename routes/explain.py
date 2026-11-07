"""Authenticated synchronous retrieval endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, require_shared_secret_auth
from config import GeminiSettings
from models import ExplainRequest, ExplainResponse
from services.retrieval import AttachmentError, AttachmentStorageError, explain

logger = logging.getLogger(__name__)


def make_explain_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post("/explain", response_model=ExplainResponse, tags=["Q&A"],
                 response_model_exclude_unset=True,
                 dependencies=[Depends(require_shared_secret_auth)])
    async def explain_endpoint(body: ExplainRequest, request: Request) -> ExplainResponse:
        if len(body.attachment_ids) > 8:
            raise HTTPException(422, "Maximum 8 attachments per request")
        database = getattr(request.app.state, "ingest_database", None)
        if not settings.use_enterprise or database is None:
            raise HTTPException(status_code=503, detail="Course material retrieval is unavailable.")
        try:
            return await explain(
                question=body.question, course_offering_id=body.course_offering_id,
                week=body.week, settings=settings, database=database, attachment_ids=body.attachment_ids,
            )
        except AttachmentError as error:
            raise HTTPException(422, str(error)) from None
        except AttachmentStorageError:
            logger.warning("retrieval.attachment_storage_unavailable")
            raise HTTPException(503, UNAVAILABLE) from None
        except SQLAlchemyError:
            logger.warning("retrieval.database_unavailable")
            raise HTTPException(status_code=503, detail="Course material retrieval is temporarily unavailable.") from None
        except RuntimeError:
            logger.warning("retrieval.embedding_unavailable")
            raise HTTPException(status_code=503, detail="Question embedding is temporarily unavailable.") from None

    return router
