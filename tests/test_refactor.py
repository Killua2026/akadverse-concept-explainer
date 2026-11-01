import main


def test_main_app_has_all_expected_routes():
    paths = [r.path for r in main.app.routes]
    assert "/ingest" in paths
    assert "/documents/{document_id}" in paths
    assert "/explain-with-grounding" in paths
    assert "/health" in paths
    assert set(main.app.openapi()["paths"]) == {
        "/ingest", "/documents/{document_id}", "/explain-with-grounding", "/health"
    }
