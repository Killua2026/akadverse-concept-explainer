"""Enterprise embeddings with explicit dimensions and no retries."""

import asyncio
import math

from google import genai
from google.genai import types

from config import (GeminiSettings, EMBEDDING_MODEL, EMBEDDING_LOCATION,
                    EMBEDDING_DIMENSIONS, EMBEDDING_BATCH_SIZE,
                    EMBEDDING_BATCH_SLEEP_SECONDS)

# Retrieval embeds up to 10 chunks concurrently within a batch via
# asyncio.gather, bounded by a semaphore. The FSE serializes /explain
# calls per student session, so the effective concurrency across the
# whole service is bounded by concurrent student requests, not by
# total chunks in flight.
#
# The 1.0s pause between batches is safe for that concurrency profile.
# Ingestion uses EMBEDDING_BATCH_SLEEP_SECONDS (8s) because 100+
# document ingestions can run concurrently and need conservative
# pacing. Do not unify the two constants.
#
# If concurrent session overlap ever triggers enterprise quota
# throttling, lower EMBEDDING_RETRIEVAL_CONCURRENCY before raising
# the sleep. Do not revert to 8s; that reintroduces the artificial
# latency this fix removes.
EMBEDDING_RETRIEVAL_CONCURRENCY = 10
EMBEDDING_RETRIEVAL_BATCH_SLEEP_SECONDS = 1.0


async def embed_chunks(chunks: list[str], settings: GeminiSettings) -> list[list[float]]:
    if not chunks:
        return []
    if not settings.use_enterprise:
        raise RuntimeError("Ingestion requires enterprise mode.")
    try:
        client = genai.Client(
            enterprise=True, project=settings.google_cloud_project,
            location=EMBEDDING_LOCATION,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        )
        async with client.aio as api:
            result: list[list[float]] = []
            for index, chunk in enumerate(chunks):
                if index and index % EMBEDDING_BATCH_SIZE == 0:
                    await asyncio.sleep(EMBEDDING_BATCH_SLEEP_SECONDS)
                response = await api.models.embed_content(
                    model=EMBEDDING_MODEL, contents=[chunk],
                    config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
                )
                embeddings = response.embeddings or []
                if len(embeddings) != 1:
                    raise ValueError("Incorrect embedding count")
                values = [float(value) for value in (embeddings[0].values or [])]
                if len(values) != EMBEDDING_DIMENSIONS or not all(map(math.isfinite, values)):
                    raise ValueError("Invalid embedding dimensions or values")
                result.append(values)
            return result
    except Exception:
        # Never expose SDK errors containing authentication or request data.
        raise RuntimeError("Embedding failed; verify enterprise model access and quota.") from None


async def embed_chunks_batch(
    chunks: list[str], settings: GeminiSettings,
) -> list[list[float]]:
    if not chunks:
        return []
    if not settings.use_enterprise:
        raise RuntimeError("Retrieval embeddings require enterprise mode.")
    try:
        client = genai.Client(
            enterprise=True, project=settings.google_cloud_project,
            location=EMBEDDING_LOCATION,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        )
        async with client.aio as api:
            sem = asyncio.Semaphore(EMBEDDING_RETRIEVAL_CONCURRENCY)

            async def embed_one(chunk: str) -> list[float]:
                async with sem:
                    response = await api.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=[chunk],
                        config=types.EmbedContentConfig(
                            output_dimensionality=EMBEDDING_DIMENSIONS,
                        ),
                    )
                embeddings = response.embeddings or []
                if len(embeddings) != 1:
                    raise ValueError("Incorrect embedding count")
                values = [float(value) for value in (embeddings[0].values or [])]
                if len(values) != EMBEDDING_DIMENSIONS or not all(map(math.isfinite, values)):
                    raise ValueError("Invalid embedding dimensions or values")
                return values

            result: list[list[float]] = []
            for batch_start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                if batch_start > 0:
                    await asyncio.sleep(EMBEDDING_RETRIEVAL_BATCH_SLEEP_SECONDS)
                batch = chunks[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
                batch_results = await asyncio.gather(
                    *[embed_one(chunk) for chunk in batch],
                    return_exceptions=False,
                )
                result.extend(batch_results)
            return result
    except Exception:
        raise RuntimeError("Embedding failed; verify enterprise model access and quota.") from None
