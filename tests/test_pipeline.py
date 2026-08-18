"""
Unit tests for the RAG pipeline endpoints using FastAPI TestClient.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify that /api/v1/health returns 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_query_endpoint():
    """Verify that /api/v1/query/text returns 200 OK."""
    payload = {
        "query": "calories calculator to lose weight",
        "language": "en",
    }
    response = client.post("/api/v1/query/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "timings" in data
    assert len(data["answer"]) > 0
