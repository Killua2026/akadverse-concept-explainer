import importlib
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolate_provider(monkeypatch):
    from services import gemini_provider
    from services import model_discovery

    for name in ("LLM_PROVIDER", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(gemini_provider, "discover_enterprise_model_names", lambda *args: ["test-model"])
    monkeypatch.setattr(model_discovery, "get_valid_model_name", lambda *args: "test-model")


def _load_service():
    service = importlib.import_module("main")
    return importlib.reload(service)


def test_app_starts_when_google_api_key_is_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    service = _load_service()

    with TestClient(service.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_app_raises_when_google_api_key_is_missing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    service = _load_service()

    with pytest.raises(RuntimeError) as error_info:
        with TestClient(service.app):
            pass

    assert str(error_info.value) == (
        "GOOGLE_API_KEY environment variable is not set. "
        "Set it before starting the service."
    )


def test_enterprise_starts_without_developer_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini_enterprise")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    service = _load_service()

    async def disable_ingestion(application):
        application.state.ingest_status = "disabled (missing DATABASE_URL)"
        application.state.ingest_database = None

    configure = AsyncMock(side_effect=disable_ingestion)
    monkeypatch.setattr(service, "configure_ingestion", configure)
    with TestClient(service.app) as client:
        assert client.get("/health").json()["provider"] == "gemini_enterprise"
        assert service.app.state.worker_tasks == []
    configure.assert_awaited_once_with(service.app)


def test_enterprise_requires_project(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini_enterprise")
    service = _load_service()
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
        with TestClient(service.app):
            pass


def test_unknown_provider_fails_startup(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    service = _load_service()
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        with TestClient(service.app):
            pass
