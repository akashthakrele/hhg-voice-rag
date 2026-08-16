"""
Pydantic schemas for API request/response models and internal data structures.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────


class ChunkStrategy(str, Enum):
    """Chunking strategies tracked in metadata for analysis."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    METADATA_AWARE = "metadata_aware"


class QuerySource(str, Enum):
    """How the query was submitted."""

    VOICE = "voice"
    TEXT = "text"


# ── Chunk Metadata ──────────────────────────────────────────


class ChunkMetadata(BaseModel):
    """Metadata stored alongside each chunk in Qdrant."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_doc_id: str = ""
    strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    query_text: str = ""           # MSMARCO original query if available
    passage_id: str = ""           # MSMARCO passage ID
    language: str = "en"


# ── API Request / Response ──────────────────────────────────


class VoiceQueryRequest(BaseModel):
    """Metadata sent alongside the audio file upload."""

    language: str = Field(default="hi-IN", description="BCP-47 language code for STT")


class TextQueryRequest(BaseModel):
    """Direct text query (bypass STT)."""

    query: str = Field(..., min_length=1, max_length=2000)
    language: str = "en"


class RetrievedChunk(BaseModel):
    """A single retrieved chunk returned in the response."""

    text: str
    score: float
    metadata: ChunkMetadata


class PipelineTimings(BaseModel):
    """Per-stage latency breakdown in milliseconds."""

    stt_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    guardrail_ms: Optional[float] = None
    total_ms: float = 0.0
    exceeds_target: bool = False


class RAGResponse(BaseModel):
    """Full pipeline response."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    answer: str
    source: QuerySource = QuerySource.TEXT
    retrieved_chunks: list[RetrievedChunk] = []
    timings: PipelineTimings = Field(default_factory=PipelineTimings)
    grounded: bool = True
    guardrail_triggered: bool = False
    guardrail_reason: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"
    qdrant: dict = {}


# ── Benchmark ───────────────────────────────────────────────


class BenchmarkRequest(BaseModel):
    """Request to run N benchmark queries."""

    num_queries: int = Field(default=100, ge=1, le=1000)
    include_generation: bool = True


class BenchmarkResult(BaseModel):
    """Aggregated benchmark results."""

    num_queries: int
    include_generation: bool
    p50_ms: float
    p70_ms: float
    p100_ms: float
    mean_ms: float
    all_timings: list[PipelineTimings] = []
