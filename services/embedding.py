"""Enterprise embeddings with explicit dimensions and no retries."""

import asyncio
import math

from google import genai
from google.genai import types

from config import (GeminiSettings, EMBEDDING_MODEL, EMBEDDING_LOCATION,
                    EMBEDDING_DIMENSIONS, EMBEDDING_BATCH_SIZE,
                    EMBEDDING_BATCH_SLEEP_SECONDS)


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
