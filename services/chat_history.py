"""Postgres-backed conversation history for the explain endpoint."""

from datetime import timedelta
from typing import Literal

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.types import Interval

from config import CHAT_HISTORY_FETCH_LIMIT, CHAT_HISTORY_TTL_DAYS
from db import Database
from models import ChatMessage


async def fetch_recent_messages(
    database: Database, session_id: str, limit: int = CHAT_HISTORY_FETCH_LIMIT,
) -> list[ChatMessage]:
    async with database.sessions() as session:
        result = await session.execute(text("""
            SELECT session_id, role, content, sources_json, created_at
            FROM concept_explainer.chat_messages
            WHERE session_id = :session_id
            ORDER BY created_at DESC, id DESC
            LIMIT :limit
        """), {"session_id": session_id, "limit": limit})
        return [ChatMessage.model_validate(row) for row in reversed(result.mappings().all())]


async def persist_message(
    database: Database,
    session_id: str,
    role: Literal["user", "assistant"],
    content: str,
    sources_json: list | None = None,
    *,
    session: AsyncSession | None = None,
) -> None:
    statement = text("""
        INSERT INTO concept_explainer.chat_messages (session_id, role, content, sources_json)
        VALUES (:session_id, :role, :content, :sources_json)
    """).bindparams(bindparam("sources_json", type_=JSONB(none_as_null=True)))
    parameters = {
        "session_id": session_id, "role": role, "content": content, "sources_json": sources_json,
    }
    if session is not None:
        await session.execute(statement, parameters)
    else:
        async with database.sessions.begin() as session:
            await session.execute(statement, parameters)


async def sweep_expired_chat_messages(database: Database) -> int:
    statement = text("""
        DELETE FROM concept_explainer.chat_messages
        WHERE created_at < now() - :retention
        RETURNING id
    """).bindparams(bindparam("retention", type_=Interval()))
    async with database.sessions.begin() as session:
        result = await session.execute(statement, {"retention": timedelta(days=CHAT_HISTORY_TTL_DAYS)})
        return len(result.all())
