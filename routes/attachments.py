"""Authenticated standalone attachment endpoints."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from sqlalchemy.exc import SQLAlchemyError

from auth import require_shared_secret_auth
from config import ATTACHMENT_ALLOWED_MIME_TYPES, ATTACHMENT_MAX_SIZE_BYTES
from models import AttachmentResponse
from services.attachments import delete_attachment, get_attachment_metadata, store_attachment

logger = logging.getLogger(__name__)


def make_attachments_router() -> APIRouter:
    router = APIRouter(dependencies=[Depends(require_shared_secret_auth)], tags=["Attachments"])

    @router.post("/attachments", response_model=AttachmentResponse)
    async def upload_attachment(request: Request, file: UploadFile = File(...)) -> AttachmentResponse:
        database = getattr(request.app.state, "ingest_database", None)
        if database is None:
            raise HTTPException(503, "Attachments require a configured database.")
        settings = getattr(request.app.state, "gemini_settings", None)
        if settings is None or not settings.use_enterprise:
            raise HTTPException(503, "Attachments require LLM_PROVIDER=gemini_enterprise.")
        filename = Path(file.filename or "").name[:255]
        if not filename:
            raise HTTPException(422, "Filename is required.")
        if file.content_type not in ATTACHMENT_ALLOWED_MIME_TYPES:
            raise HTTPException(422, "Unsupported file type. Supported: PDF, DOCX, PPTX, TXT, MD, PNG, JPG, WebP.")
        content = await file.read()
        if len(content) > ATTACHMENT_MAX_SIZE_BYTES:
            raise HTTPException(413, "File exceeds 4 MiB limit.")
        if not content:
            raise HTTPException(422, "Empty file.")
        try:
            return await store_attachment(database, filename=filename, mime_type=file.content_type, content=content)
        except (SQLAlchemyError, OSError):
            logger.warning("attachments.store_failed")
            raise HTTPException(503, "Storage unavailable. Retry with backoff.") from None

    @router.get("/attachments/{attachment_id}", response_model=AttachmentResponse)
    async def attachment_metadata(attachment_id: str, request: Request) -> AttachmentResponse:
        database = getattr(request.app.state, "ingest_database", None)
        if database is None:
            raise HTTPException(503, "Attachments require a configured database.")
        settings = getattr(request.app.state, "gemini_settings", None)
        if settings is None or not settings.use_enterprise:
            raise HTTPException(503, "Attachments require LLM_PROVIDER=gemini_enterprise.")
        try:
            attachment = await get_attachment_metadata(database, attachment_id)
        except (SQLAlchemyError, OSError):
            logger.warning("attachments.get_failed")
            raise HTTPException(503, "Storage unavailable. Retry with backoff.") from None
        if attachment is None:
            raise HTTPException(404, "Attachment not found")
        return attachment

    @router.delete("/attachments/{attachment_id}", status_code=204)
    async def remove_attachment(attachment_id: str, request: Request) -> Response:
        database = getattr(request.app.state, "ingest_database", None)
        if database is None:
            raise HTTPException(503, "Attachments require a configured database.")
        settings = getattr(request.app.state, "gemini_settings", None)
        if settings is None or not settings.use_enterprise:
            raise HTTPException(503, "Attachments require LLM_PROVIDER=gemini_enterprise.")
        try:
            deleted = await delete_attachment(database, attachment_id)
        except (SQLAlchemyError, OSError):
            logger.warning("attachments.delete_failed")
            raise HTTPException(503, "Storage unavailable. Retry with backoff.") from None
        if not deleted:
            raise HTTPException(404, "Attachment not found")
        return Response(status_code=204)

    return router
