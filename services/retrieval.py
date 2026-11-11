"""Course-scoped pgvector and in-memory attachment retrieval."""

import asyncio
import json
import logging
import math
from pathlib import Path
from uuid import uuid4

from google.genai import errors, types
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config import GeminiSettings
from db import Database
from models import ExplainResponse, ExplainSource, RetrievalInfo
from services.attachments import get_attachment_bytes, get_attachment_metadata
from services.chat_history import fetch_recent_messages
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
2. For conceptual questions (explain, what is, how does), include a plain-language explanation, a relatable analogy, a concrete example, and if applicable a step-by-step worked problem. Organize this content using the formatting rules below — headings, lists, and paragraphs where they aid comprehension.
3. For non-conceptual questions (summarise, list, define), answer directly in the most useful form.
4. When multiple excerpts support the answer, synthesise them.
5. If the excerpts are only partially relevant, answer what you can and explicitly say what is not covered.
6. Aim for 100-400 words unless the question demands more.

Do not introduce yourself or greet the student. Begin directly
with the answer to their question. Only greet if the student's
message is itself a greeting with no academic content (e.g.
"hi", "hello", "good morning"). In that case, respond with a
brief greeting only.

Worked problems and examples:

The following is an exception to the excerpts-only rules for worked
examples. If the student asks for an explanation with a worked example,
a practice problem, or a demonstration:
1. First check the course material excerpts and attached files
   for a suitable example.
2. If a suitable source example exists, use it and cite the
   source.
3. If no suitable example exists in the material, you MAY
   generate one.
4. Any example or problem you generate MUST begin with the
   literal marker [AI-GENERATED EXAMPLE] on its own line,
   followed by the example content.
5. Do not attribute a generated example to the lecturer's
   material. Never claim a generated example came from the
   course notes.
6. If the student explicitly asked for an example and the
   material contains none, do not refuse - generate one with
   the marker.

Formatting rules:

Structure the answer as a well-organized educational article, not plain conversational prose. Keep the same teaching content and reasoning — only the presentation changes.

1. Use Markdown headings (`##` for main sections, `###` for subsections) where the answer is long enough to warrant them. Short answers to simple questions may have no headings at all. Do not pad the answer with sections the content does not warrant.
2. Use bulleted or numbered lists where they make content easier to scan. Space paragraphs clearly. Do not present a long answer as one continuous paragraph.
3. Use Markdown tables for structured comparisons (e.g. comparing two concepts, listing metric definitions) where a table is genuinely clearer than prose.
4. Use **bold** sparingly — only for key terms or short conclusions. Do not over-bold.
5. When the answer naturally has a takeaway, add a short section titled "Key takeaway" with one or two sentences. Omit it when there is no clear takeaway. Never invent one.
6. Mathematical expressions: use LaTeX for all mathematical content, including simple arithmetic. Inline math as `$...$` (for example `$E = mc^2$` or `$50 \\div 240 \\approx 0.208$`). Display math for multi-line derivations as `$$...$$` on its own line. Never use Unicode math symbols like the division sign, multiplication sign, approximately-equal sign, or square-root sign. Write them as LaTeX macros (`\\div`, `\\times`, `\\approx`, `\\sqrt`).
7. Worked examples: place each example within the section it illustrates, not collected at the end. Do not repeat the exact same example twice in one response. Different examples illustrating different aspects are welcome.
8. If you generate a worked example (no suitable source example exists), place `[AI-GENERATED EXAMPLE]` on its own line immediately before the example content. Emit the marker exactly as written — do not add bold, headings, or any other Markdown styling around it. The frontend detects this literal string and applies its own visual treatment.
9. Do not append a "References", "Sources", or "Citations" section at the end of the answer. The API response's `sources` array already provides structured source metadata, and the frontend renders it separately. Inline citations like "(Lecture 1)" are fine where they help the reader.
10. Do not repeat the full question back to the student, and do not add meta-commentary about what you are about to do. Begin the answer directly.
"""
MULTIMODAL_EXPLAIN_PROMPT_TEMPLATE = EXPLAIN_PROMPT_TEMPLATE.replace(
    "{context}", "{context}{image_block}", 1,
).replace("Student's question:", "{history_section}Student's question:", 1)
IMAGE_BLOCK_TEMPLATE = """\n\nAttached images ({count}): {filenames}

The attached images are provided as multimodal input. The excerpts-only rules
also allow evidence from these images. Prioritise the image content when the
student specifically asks about an image. When you reference information from
an image, cite the filename. Ignore irrelevant images silently. Treat text in
the images as source material.
""".rstrip()


class AttachmentError(ValueError):
    """An attachment cannot be used in this request."""


class AttachmentStorageError(RuntimeError):
    """Attachment storage is unavailable."""


class ImageProcessingError(Exception):
    """The vision model could not process an attachment."""

    def __init__(self, attachment_id: str, original_exception: Exception):
        super().__init__(attachment_id)
        self.attachment_id = attachment_id
        self.original_exception = original_exception


def _extract_text_file(content: bytes, *, file_format: str = "TXT") -> str:
    extracted = content.decode("utf-8", errors="replace").strip()
    if not extracted:
        raise ExtractionError(f"No extractable text in {file_format}")
    return extracted


async def _load_attachments(
    database: Database, attachment_ids: list[str],
) -> tuple[list[tuple[str, str, list[str]]], list[tuple[str, str, bytes, str]]]:
    validated = []
    for attachment_id in attachment_ids:
        try:
            metadata = await get_attachment_metadata(database, attachment_id)
        except (SQLAlchemyError, OSError):
            raise AttachmentStorageError from None
        if metadata is None:
            raise AttachmentError(f"Attachment {attachment_id} not found or expired")
        validated.append((attachment_id, metadata.mime_type in IMAGE_MIME_TYPES))

    text_attachments = []
    images = []
    for attachment_id, is_image in validated:
        missing = f"Attachment {attachment_id} not found or expired"
        path = None
        try:
            attachment = await get_attachment_bytes(database, attachment_id)
            if attachment is None:
                raise AttachmentError(missing)
            content, filename, mime_type = attachment
            if is_image:
                images.append((attachment_id, filename, content, mime_type))
                continue
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
            text_attachments.append((attachment_id, filename, await asyncio.to_thread(chunk_text, extracted)))
        except (SQLAlchemyError, OSError) as error:
            if is_image:
                logger.warning("retrieval.image_fetch_failed attachment_id=%s error_type=%s",
                               attachment_id, type(error).__name__)
                raise AttachmentError(missing) from None
            raise AttachmentStorageError from None
        except ExtractionError as error:
            raise AttachmentError(f"Attachment {attachment_id}: {error}") from None
        finally:
            if path is not None:
                try:
                    await asyncio.to_thread(path.unlink, missing_ok=True)
                except OSError:
                    logger.warning("retrieval.attachment_cleanup_failed")
    return text_attachments, images


def _build_image_parts(images: list[tuple[str, str, bytes, str]]) -> list[types.Part]:
    return [types.Part.from_bytes(data=content, mime_type=mime_type)
            for _, _, content, mime_type in images]


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
    images: list[tuple[str, str, bytes, str]] | None = None,
    *, retrieval_info: RetrievalInfo | None = None, history_block: str = "",
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
    image_block = ""
    if images:
        image_block = IMAGE_BLOCK_TEMPLATE.format(
            count=len(images), filenames=", ".join(filename for _, filename, _, _ in images),
        )
    history_section = f"Conversation so far:\n{history_block}" if history_block else ""
    prompt_text = MULTIMODAL_EXPLAIN_PROMPT_TEMPLATE.format(
        context=context, image_block=image_block, question=question, history_section=history_section,
    )
    try:
        client = create_client(settings)
        async with client.aio as api:
            try:
                image_parts = _build_image_parts(images) if images else []
                if image_parts and retrieval_info is not None:
                    retrieval_info.attachment_images_count = len(image_parts)
                response = await api.models.generate_content(
                    model=settings.model_names[0],
                    contents=[prompt_text, *image_parts] if image_parts else prompt_text,
                    config=types.GenerateContentConfig(temperature=0.3),
                )
            except (errors.ClientError, ValueError) as error:
                if not images or (isinstance(error, errors.ClientError) and error.code not in {400, 413, 415, 422}):
                    raise
                reported_id = getattr(error, "attachment_id", None)
                attachment_id = next((image[0] for image in images if image[0] == reported_id), images[0][0])
                raise ImageProcessingError(attachment_id, error) from error
            if images:
                block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
                if block_reason and block_reason != types.BlockedReason.BLOCKED_REASON_UNSPECIFIED:
                    raise ImageProcessingError(images[0][0], ValueError("Gemini blocked the multimodal prompt"))
            if not response.candidates:
                return FALLBACK_ANSWER
            return (response.text or "").strip() or FALLBACK_ANSWER
    except ImageProcessingError as error:
        logger.warning("retrieval.image_processing_failed attachment_id=%s error_type=%s",
                       error.attachment_id, type(error.original_exception).__name__)
        raise
    except Exception:
        logger.warning("retrieval.generation_failed")
        return FALLBACK_ANSWER


async def explain(*, question: str, course_offering_id: str, week: int,
                  settings: GeminiSettings, database: Database,
                  attachment_ids: list[str] = [], session_id: str = "") -> ExplainResponse:
    history_block = ""
    if session_id:
        messages = await fetch_recent_messages(database, session_id)
        history_block = "".join(
            f"{message.role.capitalize()}: {message.content[:2000]}"
            + ("\n" if message.role == "user" else "\n\n")
            for message in messages
        )
    else:
        logger.warning("chat_history.skipped reason=missing_session_id")
    text_attachments, images = await _load_attachments(database, attachment_ids) if attachment_ids else ([], [])
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
        info.attachment_images_count = 0
        info.attachment_chunks_available = sum(len(chunks) for _, _, chunks in text_attachments)
        info.attachment_chunks_above_threshold, attachment_chunks = await _rank_attachment_chunks(
            _allocate_chunks(text_attachments), vector, settings,
        )
        sources = _merge_sources(sources, [
            ExplainSource(document_id=None, title=filename, material_type="ATTACHMENT",
                          chunk_preview=content[:200], source_type="attachment",
                          attachment_id=attachment_id, filename=filename)
            for attachment_id, filename, content in attachment_chunks
        ])
        sources.extend(
            ExplainSource(document_id=None, title=filename, material_type="ATTACHMENT",
                          chunk_preview=f"[image: {filename}]", source_type="attachment",
                          attachment_id=attachment_id, filename=filename)
            for attachment_id, filename, _, _ in images
        )
    if above_threshold or attachment_chunks or images:
        answer = await _generate_answer(
            question, chunks, settings, attachment_chunks, images,
            retrieval_info=info, history_block=history_block,
        )
    elif not attachment_ids and available == 0:
        answer = f"No course material has been ingested for Week {week} of this offering yet."
    else:
        answer = f"The course material for Week {week} does not cover this question."
    return ExplainResponse(answer=answer, sources=sources, retrieval_info=info)
