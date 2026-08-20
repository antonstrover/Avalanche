from fastapi.testclient import TestClient

from avalanche.api.app import app

client = TestClient(app)


def test_health_reports_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_config_options_serves_the_resolved_config_schema():
    response = client.get("/api/config-options")
    assert response.status_code == 200
    body = response.json()
    assert "schema" in body
    assert body["schema"]["title"] == "ResolvedConfig"
    assert "seed" in body["schema"]["properties"]


def test_openapi_document_is_generated():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/api/config-options" in response.json()["paths"]
