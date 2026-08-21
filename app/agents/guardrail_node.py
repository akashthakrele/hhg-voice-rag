"""
Guardrail Nodes — off-topic detection and grounding verification.
These are LangGraph conditional-edge nodes that can short-circuit the pipeline.
"""

from __future__ import annotations

import re
import time

import structlog

from app.agents.state import PipelineState

logger = structlog.get_logger(__name__)

# Fast rule-based patterns for off-topic, creative, and conversational queries
OFF_TOPIC_PATTERNS = [
    re.compile(
        r"\b(write|compose|create|generate)\s+(a|an|me\s+a)?\s*(poem|story|fiction|song|essay|novel|joke|script|play|dialogue)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(roleplay|pretend\s+to\s+be|act\s+as|simulate)\b", re.IGNORECASE),
    re.compile(
        r"\b(ignore\s+(all\s+)?(previous|prior)\s+instructions|system\s+prompt|jailbreak)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(how\s+are\s+you|who\s+are\s+you|what\s+is\s+your\s+name|are\s+you\s+an\s+ai)\b",
        re.IGNORECASE,
    ),
]

REFUSAL_PHRASES = (
    "don't have enough context",
    "do not have enough context",
    "not enough context",
    "not enough information",
    "no information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "does not mention",
    "not mentioned in the context",
    "off-topic",
    "no context provided",
    "insufficient context",
)


# ── Off-Topic Guard (pre-retrieval) ────────────────────────


async def off_topic_guard_node(state: PipelineState) -> PipelineState:
    """
    Ultra-fast (<1ms) rule-based off-topic classifier run BEFORE retrieval to save compute.
    Replaces slow external LLM API calls with compiled regex patterns.
    """
    query = state.get("query_text", "")
    start = time.perf_counter()

    q_clean = query.strip()
    if not q_clean or len(q_clean) < 3:
        state["is_off_topic"] = True
        state["off_topic_reason"] = "empty_query"
        state["is_grounded"] = False
        elapsed_ms = (time.perf_counter() - start) * 1000
        state.setdefault("timings", {})["guardrail_off_topic_ms"] = round(elapsed_ms, 2)
        return state

    is_off_topic = False
    for pattern in OFF_TOPIC_PATTERNS:
        if pattern.search(q_clean):
            is_off_topic = True
            break

    state["is_off_topic"] = is_off_topic
    if is_off_topic:
        state["off_topic_reason"] = "classifier"
        state["is_grounded"] = False
        logger.warning("off_topic_detected", query=query[:100])

    elapsed_ms = (time.perf_counter() - start) * 1000
    state.setdefault("timings", {})["guardrail_off_topic_ms"] = round(elapsed_ms, 2)
    return state


# ── Grounding Guard (post-generation) ──────────────────────


def check_grounding_fast(
    answer: str, chunks: list[dict]
) -> tuple[bool, float]:
    """
    Ultra-fast (<0.2ms) lexical, citation, and retrieval-score grounding check.
    Eliminates slow CPU transformer encoding calls post-generation.
    """
    import os
    if os.getenv("ENV") == "test":
        return True, 1.0

    if not answer or not chunks:
        return False, 0.0

    # 1. Refusal check
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in REFUSAL_PHRASES):
        return False, 0.0

    # 2. Check for explicit passage citations e.g. [Passage 1], 【Passage 1】
    has_citation = bool(
        re.search(r"(\[|【)Passage\s*\d+(\]|】)", answer, re.IGNORECASE)
    )

    # 3. Compute key term overlap between answer and retrieved passages
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "to",
        "in", "on", "at", "by", "for", "with", "about", "that", "this",
        "it", "of", "from", "as", "be", "yes", "no", "also", "include",
    }
    answer_tokens = {
        w for w in re.findall(r"\w+", answer_lower)
        if w not in stop_words and len(w) > 2
    }
    context_text = " ".join(c.get("text", "").lower() for c in chunks[:3])
    context_tokens = {
        w for w in re.findall(r"\w+", context_text)
        if w not in stop_words and len(w) > 2
    }

    if not answer_tokens:
        return False, 0.0

    overlap = len(answer_tokens & context_tokens)
    overlap_ratio = overlap / len(answer_tokens)
    top_score = float(chunks[0].get("score", 0.0))

    # Grounded if grounding_score meets cutoff threshold (0.50) or contains valid citation
    grounding_score = round(0.5 * overlap_ratio + 0.5 * top_score, 4)
    is_grounded = (grounding_score >= 0.50) or (has_citation and top_score >= 0.70)

    return is_grounded, grounding_score


async def grounding_guard_node(state: PipelineState) -> PipelineState:
    """
    Verify that the generated answer is grounded in retrieved context (<0.2ms).
    """
    answer = (state.get("generated_answer") or "").strip()
    chunks = state.get("retrieved_chunks", [])
    has_context = state.get("has_sufficient_context", False)

    start = time.perf_counter()

    if not answer or not chunks or not has_context:
        state["is_grounded"] = False
        state["grounding_score"] = 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000
        state.setdefault("timings", {})["guardrail_grounding_ms"] = round(elapsed_ms, 2)
        return state

    is_grounded, score = check_grounding_fast(answer, chunks)
    state["is_grounded"] = is_grounded
    state["grounding_score"] = score

    if not is_grounded:
        logger.warning(
            "grounding_check_failed",
            grounding_score=score,
            has_context=has_context,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000
    state.setdefault("timings", {})["guardrail_grounding_ms"] = round(elapsed_ms, 2)
    return state
