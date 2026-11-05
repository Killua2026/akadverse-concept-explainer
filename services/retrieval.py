"""Course-scoped pgvector retrieval and grounded answer generation."""

import json
import logging

from google.genai import types
from sqlalchemy import text

from config import GeminiSettings
from db import Database
from models import ExplainResponse, ExplainSource, RetrievalInfo
from services.embedding import embed_chunks
from services.gemini_provider import create_client

logger = logging.getLogger(__name__)
SIMILARITY_THRESHOLD = 0.4
FALLBACK_ANSWER = "I could not generate an answer from the provided material. Please try rephrasing your question."
EXPLAIN_PROMPT_TEMPLATE = """You are Stewart, AkadVerse's study assistant. Answer the student's question using ONLY the course material excerpts below.

Course material excerpts:
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


async def _generate_answer(question: str, chunks: list, settings: GeminiSettings) -> str:
    context = "\n\n".join(
        f"--- Source: {chunk['metadata']['title']} ---\n{chunk['content']}" for chunk in chunks
    )
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
                  settings: GeminiSettings, database: Database) -> ExplainResponse:
    vector = await _embed_question(question, settings)
    available, above_threshold, chunks = await _query_chunks(database, vector, course_offering_id, week)
    info = RetrievalInfo(chunks_available=available, chunks_above_threshold=above_threshold,
                         similarity_threshold=SIMILARITY_THRESHOLD)
    if available == 0:
        answer = f"No course material has been ingested for Week {week} of this offering yet."
    elif above_threshold == 0:
        answer = f"The course material for Week {week} does not cover this question."
    else:
        answer = await _generate_answer(question, chunks, settings)
    return ExplainResponse(answer=answer, sources=[
        ExplainSource(document_id=chunk["document_id"], title=chunk["metadata"]["title"],
                      material_type=chunk["metadata"]["material_type"], chunk_preview=chunk["content"][:200])
        for chunk in chunks
    ], retrieval_info=info)
