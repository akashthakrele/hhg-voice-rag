"""
Tests for API endpoints — health check, text query, and error handling.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_root(client: AsyncClient):
    """Root endpoint serves frontend HTML UI."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Voice-Enabled RAG Pipeline" in response.text or "Voice RAG Pipeline" in response.text


@pytest.mark.asyncio
async def test_static_assets(client: AsyncClient):
    """Static assets (CSS, JS) are served correctly."""
    css_res = await client.get("/style.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers.get("content-type", "")

    js_res = await client.get("/app.js")
    assert js_res.status_code == 200
    assert "javascript" in js_res.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Health endpoint returns status (may be degraded without Qdrant)."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data


@pytest.mark.asyncio
async def test_text_query_empty_body(client: AsyncClient):
    """Text query with empty body returns 422."""
    response = await client.post("/api/v1/query/text", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_text_query_requires_query_field(client: AsyncClient):
    """Text query must have a 'query' field."""
    response = await client.post(
        "/api/v1/query/text",
        json={"language": "en"},  # missing 'query'
    )
    assert response.status_code == 422
