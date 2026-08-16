"""
Generation Node — LLM answer generation via Groq API.
Uses Llama 3.3 70B for quality, falls back to 3.1 8B for speed.
"""

from __future__ import annotations

import time

import httpx
import structlog

from app.agents.state import PipelineState
from app.core.config import get_settings
from app.exceptions import GenerationError
from app.prompts.generation import GENERATION_SYSTEM_PROMPT, GENERATION_USER_TEMPLATE

logger = structlog.get_logger(__name__)


async def generation_node(state: PipelineState) -> PipelineState:
    """
    Generate an answer using Groq LLM based on retrieved context.

    Includes retry with fallback to smaller model.
    Sets `generated_answer` in state.
    """
    query = state.get("query_text", "")
    chunks = state.get("retrieved_chunks", [])

    # Check if we have sufficient context
    if not state.get("has_sufficient_context", False):
        state["generated_answer"] = (
            "I don't have enough context in the knowledge base to answer "
            "this question accurately. Please try rephrasing your question "
            "or asking about a different topic."
        )
        logger.info("generation_skipped", reason="insufficient_context")
        return state

    # Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        context_parts.append(f"[Passage {i} | relevance: {score:.3f}]\n{text}")
    context = "\n\n".join(context_parts)

    settings = get_settings()
    start = time.perf_counter()

    # Try primary model, then fallback
    models_to_try = [settings.groq_model, settings.groq_fallback_model]
    last_error: Exception | None = None

    for model in models_to_try:
        for attempt in range(settings.max_retries + 1):
            try:
                answer = await _call_groq(query, context, model, settings)
                elapsed_ms = (time.perf_counter() - start) * 1000

                state["generated_answer"] = answer
                state.setdefault("timings", {})["generation_ms"] = round(elapsed_ms, 2)

                logger.info(
                    "generation_complete",
                    model=model,
                    answer_length=len(answer),
                    elapsed_ms=round(elapsed_ms, 2),
                    attempt=attempt + 1,
                )
                return state

            except Exception as exc:
                last_error = exc
                if attempt < settings.max_retries:
                    delay = settings.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "generation_retry",
                        model=model,
                        attempt=attempt + 1,
                        delay_s=delay,
                        error=str(exc),
                    )
                    import asyncio
                    await asyncio.sleep(delay)

        logger.warning("generation_model_failed", model=model, error=str(last_error))

    state["error"] = f"Generation failed with all models: {last_error}"
    raise GenerationError(str(last_error))


async def _call_groq(query: str, context: str, model: str, settings) -> str:
    """Make the Groq API call for answer generation."""
    user_message = GENERATION_USER_TEMPLATE.format(
        context=context,
        query=query,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 512,
                "temperature": 0.1,  # Low temp for factual grounding
                "top_p": 0.9,
            },
        )
        response.raise_for_status()
        data = response.json()

    answer = data["choices"][0]["message"]["content"]
    if not answer.strip():
        raise GenerationError("Empty response from Groq")

    return answer.strip()
