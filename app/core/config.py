"""
Core configuration — single source of truth for all env vars and settings.
Uses pydantic-settings for validation + .env file loading.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── API Keys ────────────────────────────────────────────
    sarvam_api_key: str = ""
    groq_api_key: str = ""

    # ── Qdrant ──────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "msmarco_xi"

    # ── Embedding Model ────────────────────────────────────
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024

    # ── LLM ─────────────────────────────────────────────────
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"

    # ── Chunking ────────────────────────────────────────────
    chunk_size: int = 256
    chunk_overlap: int = 50

    # ── Latency ─────────────────────────────────────────────
    latency_target_ms: float = 200.0
    log_level: str = "INFO"

    # ── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Sarvam STT ──────────────────────────────────────────
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text-translate"
    sarvam_stt_language: str = "hi-IN"

    # ── Retrieval ───────────────────────────────────────────
    retrieval_top_k: int = 5
    grounding_similarity_threshold: float = 0.45

    # ── Retry ───────────────────────────────────────────────
    max_retries: int = 2
    retry_base_delay: float = 0.5


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
