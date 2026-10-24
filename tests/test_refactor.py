import main


def test_main_app_has_all_expected_routes():
    paths = [r.path for r in main.app.routes]
    assert "/upload-note" in paths
    assert "/explain-concept" in paths
    assert "/explain-with-grounding" in paths
    assert "/health" in paths
