"""Temporary attachment storage shared by all authenticated callers."""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import text

from config import ATTACHMENT_TTL_HOURS
from db import Database
from models import AttachmentResponse


async def store_attachment(
    database: Database, *, filename: str, mime_type: str, content: bytes,
) -> AttachmentResponse:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ATTACHMENT_TTL_HOURS)
    async with database.sessions() as session, session.begin():
        result = await session.execute(text("""
            INSERT INTO concept_explainer.attachments
                (filename, mime_type, size_bytes, content, expires_at)
            VALUES (:filename, :mime_type, :size_bytes, :content, :expires_at)
            RETURNING id::text AS attachment_id, filename, mime_type, size_bytes,
                      created_at, expires_at
        """), {"filename": filename, "mime_type": mime_type, "size_bytes": len(content),
               "content": content, "expires_at": expires_at})
        return AttachmentResponse(**result.mappings().one())


async def get_attachment_metadata(database: Database, attachment_id: str) -> AttachmentResponse | None:
    try:
        identifier = UUID(attachment_id)
    except ValueError:
        return None
    async with database.sessions() as session:
        result = await session.execute(text("""
            SELECT id::text AS attachment_id, filename, mime_type, size_bytes, created_at, expires_at
            FROM concept_explainer.attachments WHERE id = :id AND expires_at > now()
        """), {"id": identifier})
        row = result.mappings().first()
        return AttachmentResponse(**row) if row is not None else None


async def get_attachment_bytes(database: Database, attachment_id: str) -> tuple[bytes, str, str] | None:
    try:
        identifier = UUID(attachment_id)
    except ValueError:
        return None
    async with database.sessions() as session:
        result = await session.execute(text("""
            SELECT content, filename, mime_type FROM concept_explainer.attachments
            WHERE id = :id AND expires_at > now()
        """), {"id": identifier})
        row = result.mappings().first()
        return (bytes(row["content"]), row["filename"], row["mime_type"]) if row is not None else None


async def delete_attachment(database: Database, attachment_id: str) -> bool:
    try:
        identifier = UUID(attachment_id)
    except ValueError:
        return False
    async with database.sessions() as session, session.begin():
        result = await session.execute(text(
            "DELETE FROM concept_explainer.attachments WHERE id = :id RETURNING id"
        ), {"id": identifier})
        return result.first() is not None


async def sweep_expired_attachments(database: Database) -> int:
    async with database.sessions() as session, session.begin():
        result = await session.execute(text(
            "DELETE FROM concept_explainer.attachments WHERE expires_at < now() RETURNING id"
        ))
        return len(result.all())
