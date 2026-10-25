import importlib
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from config import GeminiSettings
from services import gemini_provider as provider
from services import model_discovery
from routes.grounding import make_grounding_router


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("LLM_PROVIDER", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_priority_list():
    assert provider.ENTERPRISE_MODEL_PRIORITY == [
        "gemini-3.1-flash-lite", "gemini-3.5-flash-lite",
        "gemini-3.5-flash", "gemini-3.8-flash",
    ]


def test_discovery_success(monkeypatch):
    names = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-image", "text-embedding", "gemini-other-flash"]
    client = SimpleNamespace(models=SimpleNamespace(list=Mock(return_value=[
        SimpleNamespace(name="publishers/google/models/" + name, supported_actions=["generateContent"])
        for name in names
    ])))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(provider.genai, "Client", constructor)
    assert not inspect.iscoroutinefunction(provider.discover_enterprise_model_names)
    assert provider.discover_enterprise_model_names("test-project", "us-central1") == [
        "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-other-flash",
    ]
    constructor.assert_called_once_with(enterprise=True, project="test-project", location="us-central1")
    client.models.list.assert_called_once_with()


def test_discovery_failure(monkeypatch, caplog):
    monkeypatch.setattr(provider.genai, "Client", Mock(side_effect=RuntimeError("private-error-text")))
    assert provider.discover_enterprise_model_names("test-project", "us-central1") == provider.ENTERPRISE_MODEL_PRIORITY
    assert "falling back to priority list; discovery failed" in caplog.text
    assert "private-error-text" not in caplog.text


@pytest.mark.parametrize("enterprise", [True, False])
def test_create_client(monkeypatch, enterprise):
    settings = GeminiSettings(google_api_key="test-key", google_cloud_project="test-project",
                              llm_provider="gemini_enterprise" if enterprise else "gemini")
    constructor = Mock()
    monkeypatch.setattr(provider.genai, "Client", constructor)
    assert provider.create_client(settings) is constructor.return_value
    expected = {"enterprise": True, "project": "test-project", "location": "global"} if enterprise else {"api_key": "test-key"}
    # Gemini 3.x models are served at global; see docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations
    constructor.assert_called_once_with(**expected)


def test_unconfigured_client():
    assert provider.create_client(GeminiSettings()) is None


def test_project_does_not_enable_enterprise(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    settings = GeminiSettings.from_env()
    assert not settings.use_enterprise
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        settings.validate()


def test_settings_cache_enterprise_discovery(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini_enterprise")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    discovery = Mock(return_value=["test-model"])
    monkeypatch.setattr(provider, "discover_enterprise_model_names", discovery)
    settings = GeminiSettings.from_env()
    settings.validate()
    assert settings.model_names == ["test-model"]
    # Gemini 3.x models are served at global; see docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations
    discovery.assert_called_once_with("test-project", "global")


@pytest.mark.parametrize("enterprise", [True, False])
@pytest.mark.anyio
async def test_grounding_selected_provider(monkeypatch, enterprise):
    settings = GeminiSettings(google_api_key="test-key", google_cloud_project="test-project",
                              llm_provider="gemini_enterprise" if enterprise else "gemini",
                              model_names=["test-model"])
    fields = {name: "test answer" for name in (
        "concept", "simple_explanation", "analogy", "example", "worked_problem"
    )}
    generate = AsyncMock(return_value=SimpleNamespace(
        text=json.dumps(fields), candidates=[SimpleNamespace(grounding_metadata=None)]
    ))
    constructor = Mock(return_value=SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))))
    monkeypatch.setattr(provider.genai, "Client", constructor)
    app = FastAPI()
    app.include_router(make_grounding_router(settings.google_api_key, settings=settings))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as session:
        response = await session.post("/explain-with-grounding", data={"question": "Recursion?"})
    assert response.status_code == 200
    assert response.json() == {**fields, "sources": []}
    await_args = generate.await_args
    assert await_args is not None
    assert await_args.kwargs["model"] == "test-model"


@pytest.mark.parametrize("enterprise", [True, False])
def test_legacy_wrapper_arguments(monkeypatch, enterprise):
    from routes import legacy
    from fastapi.testclient import TestClient
    settings = GeminiSettings(google_api_key="test-key", google_cloud_project="test-project",
                              llm_provider="gemini_enterprise" if enterprise else "gemini",
                              model_names=["test-model"])
    embeddings = Mock()
    chat = Mock()
    expected_answer = legacy.ExplanationResponse(**{name: "answer" for name in (
        "concept", "simple_explanation", "analogy", "example", "worked_problem"
    )})
    chat.return_value.with_structured_output.return_value.invoke.return_value = expected_answer
    monkeypatch.setattr(legacy, "ChatGoogleGenerativeAI", chat)
    router = legacy.make_legacy_router(settings.google_api_key, settings=settings)
    router.legacy_dependencies["embeddings"] = embeddings
    store = Mock()
    store.as_retriever.return_value.invoke.return_value = [SimpleNamespace(page_content="notes")]
    router.legacy_dependencies["vector_store"] = lambda _: store
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/explain-concept", data={"question": "Recursion?"})
    assert response.status_code == 200
    args = {"vertexai": True, "project": "test-project", "location": "global"} if enterprise else {"vertexai": False, "api_key": "test-key"}
    # Gemini 3.x models are served at global; see docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations
    embeddings.assert_called_once_with(model="models/gemini-embedding-001", **args)
    chat.assert_called_once_with(model="test-model", temperature=0.3, **args)
