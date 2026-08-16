"""
FastAPI application entrypoint.
Voice-Enabled RAG Pipeline — HH Goa Task 2.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.db import ensure_collection_exists

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO+
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    settings = get_settings()
    logger.info(
        "startup",
        qdrant_url=settings.qdrant_url,
        embedding_model=settings.embedding_model,
        groq_model=settings.groq_model,
        latency_target_ms=settings.latency_target_ms,
    )

    # Ensure Qdrant collection exists
    try:
        ensure_collection_exists()
        logger.info("qdrant_collection_ready", collection=settings.qdrant_collection)
    except Exception as exc:
        logger.warning("qdrant_init_failed", error=str(exc))

    yield

    logger.info("shutdown")


app = FastAPI(
    title="Voice-Enabled RAG Pipeline",
    description=(
        "HH Goa Task 2 — Voice input → STT → Chunking/Retrieval → "
        "Answer Generation with guardrails and latency instrumentation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins in dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router, prefix="/api/v1")


# Root redirect to docs
@app.get("/", tags=["system"])
async def root():
    """Root endpoint — redirects to API docs."""
    return {
        "message": "Voice-Enabled RAG Pipeline — HH Goa Task 2",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
