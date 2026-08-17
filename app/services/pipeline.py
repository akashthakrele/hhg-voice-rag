"""
Pipeline orchestration service — the main entry point that invokes the LangGraph.
Bridges the API layer with the agent graph.
"""

from __future__ import annotations

import time
import uuid

import structlog

from app.agents.graph import build_rag_graph
from app.agents.state import PipelineState
from app.schemas import (
    RAGResponse,
    RetrievedChunk,
    ChunkMetadata,
    PipelineTimings,
    QuerySource,
)
from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# Lazy-compiled graph singleton
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


async def run_voice_pipeline(
    audio_bytes: bytes,
    language: str = "hi-IN",
) -> RAGResponse:
    """
    Full voice pipeline: audio → STT → guardrail → retrieve → generate → grounding check.
    """
    return await _run_pipeline(
        audio_bytes=audio_bytes,
        query_text=None,
        language=language,
        source=QuerySource.VOICE,
    )


async def run_text_pipeline(
    query: str,
    language: str = "en",
) -> RAGResponse:
    """
    Text query pipeline: text → guardrail → retrieve → generate → grounding check.
    Skips STT node.
    """
    return await _run_pipeline(
        audio_bytes=None,
        query_text=query,
        language=language,
        source=QuerySource.TEXT,
    )


async def _run_pipeline(
    audio_bytes: bytes | None,
    query_text: str | None,
    language: str,
    source: QuerySource,
) -> RAGResponse:
    """
    Internal pipeline runner — invokes the LangGraph and maps state to response.
    """
    settings = get_settings()
    request_id = str(uuid.uuid4())
    overall_start = time.perf_counter()

    # Build initial state
    initial_state: PipelineState = {
        "audio_bytes": audio_bytes,
        "query_text": query_text,
        "language": language,
        "is_off_topic": False,
        "off_topic_reason": None,
        "retrieved_chunks": [],
        "has_sufficient_context": False,
        "generated_answer": None,
        "is_grounded": True,
        "grounding_score": 0.0,
        "timings": {},
        "error": None,
        "retry_count": 0,
        "transcript": None,
    }

    graph = _get_graph()

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("pipeline_error", error=str(exc), request_id=request_id)
        total_ms = (time.perf_counter() - overall_start) * 1000
        return RAGResponse(
            request_id=request_id,
            query=query_text or "(transcription failed)",
            answer=f"Pipeline error: {exc}",
            source=source,
            timings=PipelineTimings(total_ms=round(total_ms, 2), exceeds_target=True),
        )

    # Calculate total time
    total_ms = (time.perf_counter() - overall_start) * 1000
    exceeds_target = total_ms > settings.latency_target_ms

    # Build timing breakdown
    raw_timings = final_state.get("timings", {})
    timings = PipelineTimings(
        stt_ms=raw_timings.get("stt_ms"),
        retrieval_ms=raw_timings.get("retrieval_ms"),
        generation_ms=raw_timings.get("generation_ms"),
        guardrail_ms=(
            (raw_timings.get("guardrail_off_topic_ms", 0) or 0)
            + (raw_timings.get("guardrail_grounding_ms", 0) or 0)
        ) or None,
        total_ms=round(total_ms, 2),
        exceeds_target=exceeds_target,
    )

    # Determine answer and grounding status
    is_off_topic = final_state.get("is_off_topic", False)
    has_sufficient_context = final_state.get("has_sufficient_context", False)
    is_grounded = final_state.get("is_grounded", True)
    refusal_phrases = (
        "don't have enough context",
        "do not have enough context",
        "not enough context",
        "not enough information",
        "no information",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "off-topic",
    )

    if is_off_topic:
        answer = "This question appears to be off-topic for this knowledge base. Please ask a factual question."
        guardrail_triggered = True
        guardrail_reason = "off_topic"
        is_grounded = False
    elif not has_sufficient_context:
        answer = final_state.get("generated_answer", "Not enough context to answer.")
        guardrail_triggered = True
        guardrail_reason = "insufficient_context"
        is_grounded = False
    elif not is_grounded:
        answer = "The generated answer could not be verified against the source context. Please try rephrasing."
        guardrail_triggered = True
        guardrail_reason = "not_grounded"
        is_grounded = False
    else:
        raw_answer = final_state.get("generated_answer", "No answer generated.")
        answer = raw_answer
        # Check if the generated answer itself is a refusal
        if any(p in raw_answer.lower() for p in refusal_phrases):
            is_grounded = False
            guardrail_triggered = True
            guardrail_reason = "insufficient_context"
        else:
            guardrail_triggered = False
            guardrail_reason = None

    # Map retrieved chunks
    retrieved = []
    for chunk in final_state.get("retrieved_chunks", []):
        retrieved.append(
            RetrievedChunk(
                text=chunk.get("text", ""),
                score=chunk.get("score", 0.0),
                metadata=ChunkMetadata(**chunk.get("metadata", {})),
            )
        )

    if exceeds_target:
        logger.warning(
            "latency_target_exceeded",
            total_ms=round(total_ms, 2),
            target_ms=settings.latency_target_ms,
            includes_generation=timings.generation_ms is not None,
            request_id=request_id,
        )

    return RAGResponse(
        request_id=request_id,
        query=final_state.get("query_text", query_text or ""),
        answer=answer,
        source=source,
        retrieved_chunks=retrieved,
        timings=timings,
        grounded=is_grounded,
        guardrail_triggered=guardrail_triggered,
        guardrail_reason=guardrail_reason,
    )
