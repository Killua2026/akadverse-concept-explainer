"""Embedding contract without network calls or real sleeps."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from config import GeminiSettings
from services import embedding


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def sdk(monkeypatch):
    async def generate(**kwargs):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 768)])
    generate_mock = AsyncMock(side_effect=generate)
    api = SimpleNamespace(models=SimpleNamespace(embed_content=generate_mock))
    context = AsyncMock()
    context.__aenter__.return_value = api
    constructor = Mock(return_value=SimpleNamespace(aio=context))
    monkeypatch.setattr(embedding.genai, "Client", constructor)
    return constructor, generate_mock


def settings():
    return GeminiSettings(llm_provider="gemini_enterprise", google_cloud_project="test")


@pytest.mark.anyio
async def test_returns_correct_shape(sdk):
    result = await embedding.embed_chunks(["chunk"] * 10, settings())
    assert len(result) == 10
    assert all(len(row) == 768 and all(isinstance(v, float) for v in row) for row in result)


@pytest.mark.anyio
async def test_batches_and_sleeps(sdk, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(embedding.asyncio, "sleep", sleep)
    await embedding.embed_chunks(["chunk"] * 21, settings())
    assert [c.kwargs["contents"] for c in sdk[1].await_args_list] == [["chunk"]] * 21
    assert sleep.await_args_list == [call(8), call(8)]


@pytest.mark.anyio
async def test_model_dimensions_and_global(sdk):
    await embedding.embed_chunks(["chunk"], settings())
    args = sdk[1].await_args_list[0].kwargs
    assert args["model"] == "gemini-embedding-2"
    assert args["config"].output_dimensionality == 768
    options = sdk[0].call_args.kwargs
    assert options["enterprise"] is True
    assert options["location"] == "global"
    assert options["http_options"].retry_options.attempts == 1


@pytest.mark.anyio
async def test_gemini_error_fails_once(sdk):
    sdk[1].side_effect = RuntimeError("429 private-error")
    with pytest.raises(RuntimeError, match="Embedding failed"):
        await embedding.embed_chunks(["chunk"], settings())
    assert sdk[1].await_count == 1


@pytest.mark.anyio
async def test_invalid_dimensions(sdk):
    sdk[1].side_effect = None
    sdk[1].return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5])])
    with pytest.raises(RuntimeError, match="Embedding failed"):
        await embedding.embed_chunks(["chunk"], settings())


@pytest.mark.anyio
async def test_preserves_chunk_vector_order(sdk):
    chunks = ["first", "second", "third"]
    sdk[1].side_effect = [
        SimpleNamespace(embeddings=[SimpleNamespace(values=[value] * 768)])
        for value in (0.1, 0.2, 0.3)
    ]
    result = await embedding.embed_chunks(chunks, settings())
    assert [entry.kwargs["contents"] for entry in sdk[1].await_args_list] == [
        [chunk] for chunk in chunks
    ]
    assert result == [[value] * 768 for value in (0.1, 0.2, 0.3)]


@pytest.mark.anyio
@pytest.mark.parametrize("count", [0, 2])
async def test_rejects_incorrect_embedding_count(sdk, count):
    sdk[1].side_effect = None
    sdk[1].return_value = SimpleNamespace(embeddings=[
        SimpleNamespace(values=[0.5] * 768) for _ in range(count)
    ])
    with pytest.raises(RuntimeError, match="Embedding failed"):
        await embedding.embed_chunks(["chunk"], settings())


@pytest.mark.anyio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
async def test_rejects_nonfinite_values(sdk, value):
    sdk[1].side_effect = None
    sdk[1].return_value = SimpleNamespace(embeddings=[
        SimpleNamespace(values=[value] * 768)
    ])
    with pytest.raises(RuntimeError, match="Embedding failed"):
        await embedding.embed_chunks(["chunk"], settings())


@pytest.mark.anyio
async def test_batch_returns_correct_shape(sdk, monkeypatch):
    monkeypatch.setattr(embedding.asyncio, "sleep", AsyncMock())

    async def generate(**kwargs):
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 768)])

    sdk[1].side_effect = generate
    result = await embedding.embed_chunks_batch(["chunk"] * 20, settings())
    assert len(result) == 20
    assert all(len(row) == 768 for row in result)


@pytest.mark.anyio
async def test_batch_single_call_per_chunk(sdk, monkeypatch):
    monkeypatch.setattr(embedding.asyncio, "sleep", AsyncMock())
    sdk[1].side_effect = [
        SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 768)])
        for _ in range(25)
    ]
    await embedding.embed_chunks_batch(["chunk"] * 25, settings())
    assert sdk[1].await_count == 25
    assert all(len(entry.kwargs["contents"]) == 1 for entry in sdk[1].await_args_list)


@pytest.mark.anyio
async def test_batch_respects_concurrency_limit(sdk, monkeypatch):
    real_sleep = asyncio.sleep
    sleep = AsyncMock()
    monkeypatch.setattr(embedding.asyncio, "sleep", sleep)
    in_flight = 0
    peak_in_flight = 0

    async def generate(**kwargs):
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        try:
            await real_sleep(0.05)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 768)])
        finally:
            in_flight -= 1

    sdk[1].side_effect = generate
    await embedding.embed_chunks_batch(["chunk"] * 20, settings())
    assert 1 < peak_in_flight <= embedding.EMBEDDING_RETRIEVAL_CONCURRENCY
    sleep.assert_awaited_once_with(embedding.EMBEDDING_RETRIEVAL_BATCH_SLEEP_SECONDS)


@pytest.mark.anyio
async def test_batch_preserves_input_order(sdk):
    values = dict(zip(["first", "second", "third", "fourth", "fifth"],
                      [0.1, 0.2, 0.3, 0.4, 0.5]))

    async def generate(*, contents, **kwargs):
        value = values[contents[0]]
        await asyncio.sleep((0.5 - value) / 10)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[value] * 768)])

    sdk[1].side_effect = generate
    result = await embedding.embed_chunks_batch(list(values), settings())
    assert result == [[value] * 768 for value in values.values()]


@pytest.mark.anyio
async def test_batch_rejects_incorrect_response_count(sdk):
    sdk[1].side_effect = [SimpleNamespace(embeddings=[])]
    with pytest.raises(RuntimeError, match="Embedding failed"):
        await embedding.embed_chunks_batch(["chunk"], settings())
