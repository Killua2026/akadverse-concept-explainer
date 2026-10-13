import importlib

import pytest
from fastapi.testclient import TestClient



def _load_service():
    service = importlib.import_module("concept_explainer")
    return importlib.reload(service)


def test_app_starts_when_google_api_key_is_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    service = _load_service()

    with TestClient(service.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_app_raises_when_google_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    service = _load_service()

    with pytest.raises(RuntimeError) as error_info:
        with TestClient(service.app):
            pass

    assert str(error_info.value) == (
        "GOOGLE_API_KEY environment variable is not set. "
        "Set it before starting the service."
    )


def test_explain_concept_does_not_require_api_key_form_field(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    service = _load_service()

    class StubEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(service, "GoogleGenerativeAIEmbeddings", StubEmbeddings)
    monkeypatch.setattr(service, "_get_or_load_vector_store", lambda _: None)

    with TestClient(service.app) as client:
        response = client.post(
            "/explain-concept",
            data={"question": "Explain recursion."},
        )

    assert response.status_code == 400
    assert "No lecture notes found" in response.json()["detail"]
