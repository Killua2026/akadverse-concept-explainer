"""Course-scoped pgvector and in-memory attachment retrieval."""

import asyncio
import json
import logging
import math
from pathlib import Path
from uuid import uuid4

from google.genai import types
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import GeminiSettings
from db import Database
from models import ExplainResponse, ExplainSource, RetrievalInfo
from services.attachments import get_attachment_bytes, get_attachment_metadata
from services.chunking import chunk_text
from services.embedding import embed_chunks
from services.extractors import ExtractionError, extract_text
from services.gemini_provider import create_client

logger = logging.getLogger(__name__)
SIMILARITY_THRESHOLD = 0.4
ATTACHMENT_CHUNK_CAP = 100
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
TEXT_FILE_FORMATS = {"text/plain": "TXT", "text/markdown": "MD"}
FALLBACK_ANSWER = "I could not generate an answer from the provided material. Please try rephrasing your question."
EXPLAIN_PROMPT_TEMPLATE = """You are Stewart, AkadVerse's study assistant. Answer the student's question using ONLY the excerpts below. You may use both course material and attached files. If the answer comes only from an attachment, cite the attachment's filename in your response.

{context}

Student's question: {question}

Rules:
1. Answer only from the excerpts. If they do not cover the question, say so clearly - do not invent information.
2. For conceptual questions (explain, what is, how does), naturally include a plain-language explanation, a relatable analogy, a concrete example, and if applicable a step-by-step worked problem. Do NOT use section labels; weave these into natural prose.
3. For non-conceptual questions (summarise, list, define), answer directly in the most useful form.
4. When multiple excerpts support the answer, synthesise them.
5. If the excerpts are only partially relevant, answer what you can and explicitly say what is not covered.
6. Aim for 100-400 words unless the question demands more.
"""


class AttachmentError(ValueError):
    """An attachment cannot be used in this request."""


class AttachmentStorageError(RuntimeError):
    """Attachment storage is unavailable."""


def _extract_text_file(content: bytes, *, file_format: str = "TXT") -> str:
    extracted = content.decode("utf-8", errors="replace").strip()
    if not extracted:
        raise ExtractionError(f"No extractable text in {file_format}")
    return extracted


async def _load_attachment_chunks(
    database: Database, attachment_ids: list[str],
) -> list[tuple[str, str, list[str]]]:
    loaded = []
    for attachment_id in attachment_ids:
        missing = f"Attachment {attachment_id} not found or expired"
        path = None
        try:
            metadata = await get_attachment_metadata(database, attachment_id)
            if metadata is None:
                raise AttachmentError(missing)
            if metadata.mime_type in IMAGE_MIME_TYPES:
                raise AttachmentError(
                    f"Image attachments not supported in Phase 3.5b. Attachment {attachment_id} "
                    f"has mime_type {metadata.mime_type}. Image support arrives in Phase 3.5c."
                )

            attachment = await get_attachment_bytes(database, attachment_id)
            if attachment is None:
                raise AttachmentError(missing)
            content, filename, mime_type = attachment
            text_format = TEXT_FILE_FORMATS.get(mime_type)
            if text_format:
                extracted = _extract_text_file(content, file_format=text_format)
            else:
                basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
                basename = basename.encode("utf-8")[:222].decode("utf-8", errors="ignore")
                path = Path("/tmp") / f"{uuid4().hex}_{basename}"
                await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(path.write_bytes, content)
                extracted = await asyncio.to_thread(extract_text, str(path), mime_type)
            loaded.append((attachment_id, filename, await asyncio.to_thread(chunk_text, extracted)))
        except (SQLAlchemyError, OSError):
            raise AttachmentStorageError from None
        except ExtractionError as error:
            raise AttachmentError(f"Attachment {attachment_id}: {error}") from None
        finally:
            if path is not None:
                try:
                    await asyncio.to_thread(path.unlink, missing_ok=True)
                except OSError:
                    logger.warning("retrieval.attachment_cleanup_failed")
    return loaded


def _allocate_chunks(attachments: list[tuple[str, str, list[str]]]) -> list[tuple[str, str, str]]:
    counts = [len(chunks) for _, _, chunks in attachments]
    total = sum(counts)
    allocations = counts.copy()
    if total > ATTACHMENT_CHUNK_CAP:
        shares = [ATTACHMENT_CHUNK_CAP * count / total for count in counts]
        allocations = [min(count, max(1, math.floor(share))) for count, share in zip(counts, shares)]
        allocated = sum(allocations)
        while allocated < ATTACHMENT_CHUNK_CAP:
            index = max(
                (i for i, count in enumerate(counts) if allocations[i] < count),
                key=lambda i: shares[i] - allocations[i], default=None,
            )
            if index is None:
                break
            allocations[index] += 1
            allocated += 1
        while allocated > ATTACHMENT_CHUNK_CAP:
            index = min(
                (i for i, allocation in enumerate(allocations) if allocation > 1),
                key=lambda i: (allocations[i] - shares[i], -i),
            )
            allocations[index] -= 1
            allocated -= 1
    return [
        (attachment_id, filename, chunk)
        for (attachment_id, filename, chunks), allocation in zip(attachments, allocations)
        for chunk in chunks[:allocation]
    ]


async def _rank_attachment_chunks(
    chunks: list[tuple[str, str, str]], vector: list[float], settings: GeminiSettings,
) -> tuple[int, list[tuple[str, str, str]]]:
    if not chunks:
        return 0, []
    embeddings = await embed_chunks([content for _, _, content in chunks], settings)
    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count mismatch")
    question_norm = math.hypot(*vector)
    scored = []
    for chunk, embedding in zip(chunks, embeddings):
        norm = question_norm * math.hypot(*embedding)
        similarity = math.fsum(a * b for a, b in zip(vector, embedding)) / norm if norm else 0.0
        if similarity >= SIMILARITY_THRESHOLD:
            scored.append((1 - similarity, chunk))
    scored.sort(key=lambda hit: hit[0])
    return len(scored), [chunk for _, chunk in scored[:8]]


def _merge_sources(
    course_sources: list[ExplainSource], attachment_sources: list[ExplainSource],
) -> list[ExplainSource]:
    return [
        source.model_copy(update={"source_type": "course_material", "attachment_id": None, "filename": None})
        for source in course_sources
    ] + attachment_sources


async def _embed_question(question: str, settings: GeminiSettings) -> list[float]:
    return (await embed_chunks([question], settings))[0]


async def _query_chunks(database: Database, query_vector: list[float],
                        course_offering_id: str, week: int) -> tuple[int, int, list]:
    parameters = {"query": json.dumps(query_vector), "coid": course_offering_id, "week": week}
    async with database.sessions() as session:
        counts = (await session.execute(text("""
            SELECT count(*) AS available,
                   count(*) FILTER (WHERE embedding <=> CAST(:query AS vector) <= 0.6) AS above_threshold
            FROM concept_explainer.chunks
            WHERE metadata->>'course_offering_id' = :coid
              AND (metadata->'weeks') @> to_jsonb(CAST(:week AS int))
        """), parameters)).mappings().one()
        chunks = (await session.execute(text("""
            SELECT document_id, content, metadata FROM concept_explainer.chunks
            WHERE metadata->>'course_offering_id' = :coid
              AND (metadata->'weeks') @> to_jsonb(CAST(:week AS int))
              AND embedding <=> CAST(:query AS vector) <= 0.6
            ORDER BY embedding <=> CAST(:query AS vector) LIMIT 8
        """), parameters)).mappings().all()
    return counts["available"], counts["above_threshold"], chunks


async def _generate_answer(
    question: str, chunks: list, settings: GeminiSettings,
    attachment_chunks: list[tuple[str, str, str]] | None = None,
) -> str:
    context = "\n\n".join(
        f"--- Source: {chunk['metadata']['title']} ---\n{chunk['content']}" for chunk in chunks
    )
    if attachment_chunks is not None:
        sections = []
        if context:
            sections.append(f"--- Course material ---\n{context}")
        if attachment_chunks:
            excerpts = "\n\n".join(
                f"--- Attachment: {filename} ---\n{content}" for _, filename, content in attachment_chunks
            )
            sections.append(f"--- Attached files ---\n{excerpts}")
        context = "\n\n".join(sections)
    try:
        client = create_client(settings)
        async with client.aio as api:
            response = await api.models.generate_content(
                model=settings.model_names[0],
                contents=EXPLAIN_PROMPT_TEMPLATE.format(context=context, question=question),
                config=types.GenerateContentConfig(temperature=0.3),
            )
            if not response.candidates:
                return FALLBACK_ANSWER
            return (response.text or "").strip() or FALLBACK_ANSWER
    except Exception:
        logger.warning("retrieval.generation_failed")
        return FALLBACK_ANSWER


async def explain(*, question: str, course_offering_id: str, week: int,
                  settings: GeminiSettings, database: Database,
                  attachment_ids: list[str] = []) -> ExplainResponse:
    attachments = await _load_attachment_chunks(database, attachment_ids) if attachment_ids else []
    vector = await _embed_question(question, settings)
    available, above_threshold, chunks = await _query_chunks(database, vector, course_offering_id, week)
    info = RetrievalInfo(chunks_available=available, chunks_above_threshold=above_threshold,
                         similarity_threshold=SIMILARITY_THRESHOLD)
    sources = [
        ExplainSource(document_id=chunk["document_id"], title=chunk["metadata"]["title"],
                      material_type=chunk["metadata"]["material_type"], chunk_preview=chunk["content"][:200])
        for chunk in chunks
    ]
    attachment_chunks = None
    if attachment_ids:
        info.attachment_chunks_available = sum(len(chunks) for _, _, chunks in attachments)
        info.attachment_chunks_above_threshold, attachment_chunks = await _rank_attachment_chunks(
            _allocate_chunks(attachments), vector, settings,
        )
        sources = _merge_sources(sources, [
            ExplainSource(document_id=None, title=filename, material_type="ATTACHMENT",
                          chunk_preview=content[:200], source_type="attachment",
                          attachment_id=attachment_id, filename=filename)
            for attachment_id, filename, content in attachment_chunks
        ])
    if above_threshold or attachment_chunks:
        answer = await _generate_answer(question, chunks, settings, attachment_chunks)
    elif not attachment_ids and available == 0:
        answer = f"No course material has been ingested for Week {week} of this offering yet."
    else:
        answer = f"The course material for Week {week} does not cover this question."
    return ExplainResponse(answer=answer, sources=sources, retrieval_info=info)
