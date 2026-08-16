"""
Guardrail Nodes — off-topic detection and grounding verification.
These are LangGraph conditional-edge nodes that can short-circuit the pipeline.
"""

from __future__ import annotations

import time

import httpx
import numpy as np
import structlog

from app.agents.state import PipelineState
from app.agents.retrieval_node import get_embedder
from app.core.config import get_settings
from app.exceptions import OffTopicError, GroundingError, InsufficientContextError
from app.prompts.guardrails import (
    OFF_TOPIC_SYSTEM_PROMPT,
    OFF_TOPIC_USER_TEMPLATE,
)

logger = structlog.get_logger(__name__)


# ── Off-Topic Guard (pre-retrieval) ────────────────────────


async def off_topic_guard_node(state: PipelineState) -> PipelineState:
    """
    Cheap off-topic classifier run BEFORE retrieval to save compute.

    Uses Groq (fast inference) for a single-token classification.
    Falls back to "RELEVANT" if the classifier itself fails.
    """
    query = state.get("query_text", "")
    if not query:
        state["is_off_topic"] = True
        state["off_topic_reason"] = "empty_query"
        return state

    settings = get_settings()
    start = time.perf_counter()

    try:
        classification = await _classify_off_topic(query, settings)
        is_off_topic = classification.strip().upper() == "OFF_TOPIC"

        state["is_off_topic"] = is_off_topic
        if is_off_topic:
            state["off_topic_reason"] = "classifier"
            logger.warning("off_topic_detected", query=query[:100])

    except Exception as exc:
        # Fail open — don't block queries if classifier is down
        logger.warning("off_topic_classifier_failed", error=str(exc))
        state["is_off_topic"] = False

    elapsed_ms = (time.perf_counter() - start) * 1000
    state.setdefault("timings", {})["guardrail_off_topic_ms"] = round(elapsed_ms, 2)
    return state


async def _classify_off_topic(query: str, settings) -> str:
    """Call Groq for off-topic classification (single token response)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.groq_fallback_model,  # Use smaller model for speed
                "messages": [
                    {"role": "system", "content": OFF_TOPIC_SYSTEM_PROMPT},
                    {"role": "user", "content": OFF_TOPIC_USER_TEMPLATE.format(query=query)},
                ],
                "max_tokens": 5,
                "temperature": 0.0,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ── Grounding Guard (post-generation) ──────────────────────


async def grounding_guard_node(state: PipelineState) -> PipelineState:
    """
    Verify that the generated answer is grounded in retrieved context.

    Uses cosine similarity between answer embedding and context embeddings.
    Threshold-based: if similarity < threshold, refuse the answer.
    """
    answer = state.get("generated_answer", "")
    chunks = state.get("retrieved_chunks", [])

    if not answer or not chunks:
        state["is_grounded"] = False
        state["grounding_score"] = 0.0
        return state

    settings = get_settings()
    start = time.perf_counter()

    try:
        embedder = get_embedder()

        # Embed the answer
        answer_vec = embedder.encode(
            f"passage: {answer}", normalize_embeddings=True
        )

        # Embed concatenated context
        context_text = " ".join(c["text"] for c in chunks[:3])
        context_vec = embedder.encode(
            f"passage: {context_text}", normalize_embeddings=True
        )

        # Cosine similarity (vectors are normalized, so dot product = cosine)
        similarity = float(np.dot(answer_vec, context_vec))

        state["is_grounded"] = similarity >= settings.grounding_similarity_threshold
        state["grounding_score"] = round(similarity, 4)

        if not state["is_grounded"]:
            logger.warning(
                "grounding_check_failed",
                similarity=similarity,
                threshold=settings.grounding_similarity_threshold,
            )

    except Exception as exc:
        logger.error("grounding_check_error", error=str(exc))
        # Fail open — allow the answer through if grounding check itself errors
        state["is_grounded"] = True
        state["grounding_score"] = -1.0

    elapsed_ms = (time.perf_counter() - start) * 1000
    state.setdefault("timings", {})["guardrail_grounding_ms"] = round(elapsed_ms, 2)
    return state
