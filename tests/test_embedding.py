"""Embedding contract without network calls or real sleeps."""

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
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 768) for _ in kwargs["contents"]])
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
    assert [len(c.kwargs["contents"]) for c in sdk[1].await_args_list] == [10, 10, 1]
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
