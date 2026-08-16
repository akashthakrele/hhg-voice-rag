"""
STT Node — Speech-to-text via Sarvam AI API.
Includes retry logic with exponential backoff.
"""

from __future__ import annotations

import base64
import time

import httpx
import structlog

from app.agents.state import PipelineState
from app.core.config import get_settings
from app.exceptions import STTError

logger = structlog.get_logger(__name__)


async def stt_node(state: PipelineState) -> PipelineState:
    """
    Transcribe audio bytes to text using Sarvam AI STT API.

    If `query_text` is already set (text query path), this node is a no-op.
    Retries up to max_retries on failure with exponential backoff.
    """
    # Skip if text query was provided directly
    if state.get("query_text"):
        logger.info("stt_skipped", reason="text_query_provided")
        return state

    audio_bytes = state.get("audio_bytes")
    if not audio_bytes:
        state["error"] = "No audio or text query provided"
        return state

    settings = get_settings()
    start = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(settings.max_retries + 1):
        try:
            transcript = await _call_sarvam_stt(audio_bytes, settings)
            elapsed_ms = (time.perf_counter() - start) * 1000

            state["transcript"] = transcript
            state["query_text"] = transcript
            state.setdefault("timings", {})["stt_ms"] = round(elapsed_ms, 2)

            logger.info(
                "stt_success",
                transcript_length=len(transcript),
                elapsed_ms=round(elapsed_ms, 2),
                attempt=attempt + 1,
            )
            return state

        except Exception as exc:
            last_error = exc
            if attempt < settings.max_retries:
                delay = settings.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "stt_retry",
                    attempt=attempt + 1,
                    delay_s=delay,
                    error=str(exc),
                )
                import asyncio
                await asyncio.sleep(delay)

    state["error"] = f"STT failed after {settings.max_retries + 1} attempts: {last_error}"
    raise STTError(str(last_error))


async def _call_sarvam_stt(audio_bytes: bytes, settings) -> str:
    """Make the actual Sarvam STT API call."""
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "input": audio_b64,
        "config": {
            "language": {"sourceLanguage": settings.sarvam_stt_language},
            "audioFormat": "wav",
            "encoding": "base64",
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            settings.sarvam_stt_url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "API-Subscription-Key": settings.sarvam_api_key,
            },
        )
        response.raise_for_status()
        data = response.json()

    transcript = data.get("output", [{}])[0].get("source", "")
    if not transcript:
        # Try alternate response format
        transcript = data.get("transcript", data.get("text", ""))

    if not transcript:
        raise STTError("Empty transcript returned from Sarvam API")

    return transcript.strip()
