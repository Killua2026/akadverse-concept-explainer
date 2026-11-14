"""Authenticated synchronous retrieval endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.exc import SQLAlchemyError

from auth import UNAVAILABLE, require_shared_secret_auth
from config import GeminiSettings
from models import ExplainRequest, ExplainResponse, ExplainSource
from services.chat_history import persist_message
from services.retrieval import AttachmentError, AttachmentStorageError, ImageProcessingError, explain

logger = logging.getLogger(__name__)


def make_explain_router(settings: GeminiSettings) -> APIRouter:
    router = APIRouter()

    @router.post("/explain", response_model=ExplainResponse, tags=["Q&A"],
                 dependencies=[Depends(require_shared_secret_auth)])
    async def explain_endpoint(body: ExplainRequest, request: Request) -> ExplainResponse:
        if len(body.attachment_ids) > 8:
            raise HTTPException(422, "Maximum 8 attachments per request")
        database = getattr(request.app.state, "ingest_database", None)
        if not settings.use_enterprise or database is None:
            raise HTTPException(status_code=503, detail="Course material retrieval is unavailable.")
        try:
            response = await explain(
                question=body.question, course_offering_id=body.course_offering_id,
                week=body.week, settings=settings, database=database, attachment_ids=body.attachment_ids,
                session_id=body.session_id, concept_id=body.concept_id,
                concept_description=body.concept_description,
            )
            marker = "[AI-GENERATED EXAMPLE]"
            _, found, following = response.answer.partition(marker)
            if found:
                preview = " ".join(following.split())[:200].rstrip()
                response.sources.append(ExplainSource(
                    source_type="ai_generated", document_id=None, title="AI-generated example",
                    material_type="AI_GENERATED", chunk_preview=f"{marker} {preview}".rstrip(),
                    attachment_id=None, filename=None,
                ))
            if body.session_id:
                async with database.sessions.begin() as session:
                    await persist_message(database, body.session_id, "user", body.question, session=session)
                    await persist_message(
                        database, body.session_id, "assistant", response.answer,
                        [source.model_dump() for source in response.sources], session=session,
                    )
            return response
        except AttachmentError as error:
            raise HTTPException(422, str(error)) from None
        except ImageProcessingError as error:
            raise HTTPException(
                422, f"Image processing failed. Attachment {error.attachment_id} could not be read by the vision model.",
            ) from None
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
