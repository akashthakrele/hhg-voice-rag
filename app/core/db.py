"""
Database client initialization — Qdrant vector store connection.
Lazy-init pattern: call get_qdrant_client() to get the singleton.
"""

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton Qdrant client instance."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(url=settings.qdrant_url, timeout=10)
    return _client


async def check_qdrant_health() -> dict:
    """
    Health-check probe for Qdrant.
    Returns {"status": "healthy", ...} or {"status": "unhealthy", "error": ...}.
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        return {
            "status": "healthy",
            "collections_count": len(collections.collections),
            "collections": [c.name for c in collections.collections],
        }
    except (UnexpectedResponse, Exception) as exc:
        return {"status": "unhealthy", "error": str(exc)}


def ensure_collection_exists() -> None:
    """
    Create the MSMARCO-XI collection if it doesn't exist yet.
    Uses cosine distance for multilingual-e5-large embeddings.
    """
    from qdrant_client.http.models import Distance, VectorParams

    settings = get_settings()
    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )


def recreate_collection(collection_name: str | None = None) -> None:
    """
    Delete and recreate the collection for fresh ingests.
    """
    from qdrant_client.http.models import Distance, VectorParams

    settings = get_settings()
    client = get_qdrant_client()
    col_name = collection_name or settings.qdrant_collection

    existing = [c.name for c in client.get_collections().collections]
    if col_name in existing:
        client.delete_collection(collection_name=col_name)

    client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(
            size=settings.embedding_dimension,
            distance=Distance.COSINE,
        ),
    )
