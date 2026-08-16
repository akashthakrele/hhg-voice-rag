"""
Custom exceptions and retry/error-recovery handlers.
All pipeline-specific errors are defined here for consistent handling.
"""

from __future__ import annotations

from typing import Optional


class RAGPipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str, stage: str = "unknown", retryable: bool = False):
        self.stage = stage
        self.retryable = retryable
        super().__init__(message)


class STTError(RAGPipelineError):
    """Speech-to-text transcription failed."""

    def __init__(self, message: str = "STT transcription failed"):
        super().__init__(message, stage="stt", retryable=True)


class RetrievalError(RAGPipelineError):
    """Vector DB retrieval failed."""

    def __init__(self, message: str = "Retrieval failed"):
        super().__init__(message, stage="retrieval", retryable=True)


class GenerationError(RAGPipelineError):
    """LLM generation failed."""

    def __init__(self, message: str = "LLM generation failed"):
        super().__init__(message, stage="generation", retryable=True)


class GuardrailTriggered(RAGPipelineError):
    """A guardrail check blocked the response."""

    def __init__(
        self,
        message: str = "Response blocked by guardrail",
        reason: Optional[str] = None,
    ):
        self.reason = reason or message
        super().__init__(message, stage="guardrail", retryable=False)


class OffTopicError(GuardrailTriggered):
    """Query was classified as off-topic."""

    def __init__(self):
        super().__init__(
            message="Query is off-topic for this knowledge base",
            reason="off_topic",
        )


class GroundingError(GuardrailTriggered):
    """Generated answer is not grounded in retrieved context."""

    def __init__(self, similarity_score: float, threshold: float):
        super().__init__(
            message=(
                f"Answer not grounded in context "
                f"(similarity={similarity_score:.3f}, threshold={threshold:.3f})"
            ),
            reason="not_grounded",
        )
        self.similarity_score = similarity_score
        self.threshold = threshold


class InsufficientContextError(GuardrailTriggered):
    """Not enough relevant context to answer the query."""

    def __init__(self):
        super().__init__(
            message="Not enough context to answer this query",
            reason="insufficient_context",
        )


class AudioValidationError(RAGPipelineError):
    """Audio file failed validation."""

    def __init__(self, message: str = "Invalid audio file"):
        super().__init__(message, stage="audio_validation", retryable=False)
