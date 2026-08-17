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
        extra="ignore",
    )

    # ── API Keys ────────────────────────────────────────────
    sarvam_api_key: str = ""
    groq_api_key: str = ""
    hf_token: str = ""

    # ── Qdrant ──────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "msmarco_xi"

    # ── Embedding Model ────────────────────────────────────
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024

    # ── LLM ─────────────────────────────────────────────────
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"

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


import os

@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    settings = Settings()
    if settings.hf_token and settings.hf_token.strip():
        token_clean = settings.hf_token.strip()
        os.environ["HF_TOKEN"] = token_clean
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token_clean
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token_clean
    return settings
