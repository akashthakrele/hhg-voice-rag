"""
API routes — FastAPI endpoints for voice upload, text query, health, benchmark, and ingestion.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from app.core.db import check_qdrant_health, recreate_collection
from app.exceptions import AudioValidationError, RAGPipelineError
from app.schemas import (
    BenchmarkRequest,
    BenchmarkResult,
    HealthResponse,
    RAGResponse,
    TextQueryRequest,
)
from app.services.benchmark import run_benchmark, timings_to_csv
from app.services.ingestion import ingest_to_qdrant
from app.services.pipeline import run_text_pipeline, run_voice_pipeline
from app.utils.audio import preprocess_audio, validate_audio_size

router = APIRouter()


# ── Health ──────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """System health check — verifies API and Qdrant connectivity."""
    qdrant_status = await check_qdrant_health()
    overall = "healthy" if qdrant_status["status"] == "healthy" else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
    )


# ── Voice Query ─────────────────────────────────────────────


@router.post("/query/voice", response_model=RAGResponse, tags=["query"])
async def voice_query(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, WebM)"),
    language: str = "hi-IN",
):
    """
    Voice-enabled RAG query.

    Upload an audio file → STT → retrieval → generation → answer.
    """
    try:
        audio_bytes = await file.read()
        validate_audio_size(audio_bytes)
        processed_audio = await preprocess_audio(audio_bytes, file.filename or "audio.wav")

        response = await run_voice_pipeline(processed_audio, language)
        return response

    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RAGPipelineError as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline error ({exc.stage}): {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ── Text Query ──────────────────────────────────────────────


@router.post("/query/text", response_model=RAGResponse, tags=["query"])
async def text_query(request: TextQueryRequest):
    """
    Text-based RAG query (bypass STT).

    Submit a text query → retrieval → generation → answer.
    """
    try:
        response = await run_text_pipeline(request.query, request.language)
        return response

    except RAGPipelineError as exc:
        raise HTTPException(status_code=502, detail=f"Pipeline error ({exc.stage}): {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ── Benchmark ───────────────────────────────────────────────


@router.post("/benchmark", response_model=BenchmarkResult, tags=["benchmark"])
async def benchmark_endpoint(request: BenchmarkRequest):
    """
    Run N test queries and return P50/P70/P100 latency numbers.

    Target: full pipeline < 200ms.
    Logs clearly whether generation step is included/excluded from numbers.
    """
    result = await run_benchmark(request.num_queries, request.include_generation)
    return result


@router.post("/benchmark/csv", tags=["benchmark"])
async def benchmark_csv_endpoint(request: BenchmarkRequest):
    """
    Run benchmark and return results as downloadable CSV.
    """
    result = await run_benchmark(request.num_queries, request.include_generation)
    csv_content = timings_to_csv(result)

    return PlainTextResponse(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=benchmark_results.csv"},
    )


# ── Ingestion ───────────────────────────────────────────────


@router.post("/ingest", tags=["admin"])
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    max_records: int = 1000,
    language: str = "en",
    strategy: str = "metadata_aware",
    clear_existing: bool = False,
):
    """
    Trigger MSMARCO-XI data ingestion into Qdrant.

    Runs in background. Use /health to check if collection is populated.
    Pass clear_existing=True to wipe and recreate the collection first.
    """
    background_tasks.add_task(
        ingest_to_qdrant,
        strategy=strategy,
        max_records=max_records,
        language=language,
        clear_existing=clear_existing,
    )

    return {
        "message": "Ingestion started in background",
        "max_records": max_records,
        "language": language,
        "strategy": strategy,
        "clear_existing": clear_existing,
    }


@router.post("/collection/clear", tags=["admin"])
async def clear_collection_endpoint():
    """
    Clear and recreate the Qdrant collection for fresh ingestion.
    """
    recreate_collection()
    return {"message": "Collection cleared and recreated successfully"}
