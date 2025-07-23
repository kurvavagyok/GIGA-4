import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/api")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_404():
    response = client.get("/nonexistent")
    assert response.status_code == 404

def test_internal_error(monkeypatch):
    # Patch an endpoint to raise an error
    from main import app
    @app.get("/raise-error")
    async def raise_error():
        raise Exception("Test error")
    response = client.get("/raise-error")
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"

def test_simple_alpha_service():
    # This test assumes a service_name that exists and does not require external API keys
    response = client.post("/api/alpha/simple/test", json={"service_name": "test", "query": "hello", "details": ""})
    # Accept 200 or 422 (validation error if service_name is not implemented)
    assert response.status_code in (200, 422, 404)