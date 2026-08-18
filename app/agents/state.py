"""
LangGraph shared state — the TypedDict that flows through all nodes.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Fields are added/mutated by each node as the pipeline progresses.
    """

    # ── Input ───────────────────────────────────────────────
    audio_bytes: bytes | None          # Raw audio (voice path)
    query_text: str | None             # Transcribed or direct text query
    language: str                      # BCP-47 language code

    # ── STT ─────────────────────────────────────────────────
    transcript: str | None             # STT output

    # ── Guardrail: off-topic ────────────────────────────────
    is_off_topic: bool
    off_topic_reason: str | None

    # ── Retrieval ───────────────────────────────────────────
    retrieved_chunks: list[dict[str, Any]]  # [{text, score, metadata}, ...]
    has_sufficient_context: bool

    # ── Generation ──────────────────────────────────────────
    generated_answer: str | None

    # ── Guardrail: grounding ────────────────────────────────
    is_grounded: bool
    grounding_score: float

    # ── Metadata ────────────────────────────────────────────
    timings: dict[str, float]             # stage_name → ms
    error: str | None
    retry_count: int
