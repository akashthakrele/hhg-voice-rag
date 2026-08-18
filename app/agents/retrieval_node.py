"""
Retrieval Node — embed query + vector search against Qdrant.
Uses multilingual-e5-large for query embedding.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from sentence_transformers import SentenceTransformer

from app.agents.state import PipelineState
from app.core.config import get_settings
from app.core.db import get_qdrant_client
from app.exceptions import RetrievalError

logger = structlog.get_logger(__name__)

# Lazy-loaded singleton
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Get or initialize the embedding model (cached)."""
    global _embedder
    if _embedder is None:
        settings = get_settings()
        logger.info("loading_embedding_model", model=settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


async def retrieval_node(state: PipelineState) -> PipelineState:
    """
    Embed the query and retrieve top-K chunks from Qdrant.

    Sets `retrieved_chunks` and `has_sufficient_context` in state.
    """
    query = state.get("query_text", "")
    if not query:
        state["error"] = "No query text for retrieval"
        state["has_sufficient_context"] = False
        return state

    settings = get_settings()
    start = time.perf_counter()

    try:
        # multilingual-e5-large expects "query: " prefix for queries
        prefixed_query = f"query: {query}"

        embedder = get_embedder()
        query_vector = embedder.encode(prefixed_query, normalize_embeddings=True).tolist()

        client = get_qdrant_client()
        results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=settings.retrieval_top_k,
            with_payload=True,
        )

        chunks: list[dict[str, Any]] = []
        for hit in results:
            chunks.append({
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "metadata": {
                    k: v for k, v in hit.payload.items() if k != "text"
                },
            })

        elapsed_ms = (time.perf_counter() - start) * 1000

        state["retrieved_chunks"] = chunks
        state["has_sufficient_context"] = len(chunks) > 0 and chunks[0]["score"] > 0.3
        state.setdefault("timings", {})["retrieval_ms"] = round(elapsed_ms, 2)

        logger.info(
            "retrieval_complete",
            num_chunks=len(chunks),
            top_score=chunks[0]["score"] if chunks else 0.0,
            elapsed_ms=round(elapsed_ms, 2),
        )

        # Explicit insufficient context fallback
        if not state["has_sufficient_context"]:
            logger.warning(
                "insufficient_context",
                num_chunks=len(chunks),
                top_score=chunks[0]["score"] if chunks else 0.0,
            )

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        state.setdefault("timings", {})["retrieval_ms"] = round(elapsed_ms, 2)
        state["error"] = f"Retrieval failed: {exc}"
        raise RetrievalError(str(exc))

    return state
