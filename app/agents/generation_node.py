"""
Generation Node — LLM answer generation using Groq cloud API.
Uses lru_cache for repeated queries and MockLLM for CI testing.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

import structlog
from dotenv import load_dotenv

from app.agents.state import PipelineState
from app.prompts.generation import GENERATION_SYSTEM_PROMPT, GENERATION_USER_TEMPLATE

load_dotenv()
logger = structlog.get_logger(__name__)


# ── LLM Client Setup ──────────────────────────────────────────

_ENV = os.getenv("ENV", "production")


class _MockLLM:
    """Lightweight mock for CI / unit-test environments."""

    def create(self, *, messages, model, temperature, max_tokens):  # noqa: ARG002
        class _Choice:
            class _Msg:
                content = "Mocked answer for CI testing."
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        return _Resp()


if _ENV == "test":
    # CI / pytest – never call a real API
    class _MockCompletions:
        @staticmethod
        def create(*, messages, model, temperature, max_tokens=100):  # noqa: ARG004
            return _MockLLM().create(
                messages=messages, model=model,
                temperature=temperature, max_tokens=max_tokens,
            )

    class _MockChat:
        completions = _MockCompletions()

    class _MockClient:
        chat = _MockChat()

    _client = _MockClient()
else:
    from groq import Groq
    _api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    _client = Groq(api_key=_api_key)


def get_llm():
    """Return the Groq client (or mock). Used by main.py startup warmup."""
    return _client


# ── Cached Generation ─────────────────────────────────────────

@lru_cache(maxsize=1024)
def _cached_groq_generation(query: str, context: str) -> str:
    """Call Groq API with caching on (query, context) pairs."""
    user_prompt = GENERATION_USER_TEMPLATE.format(context=context, query=query)
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

    response = _client.chat.completions.create(
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
        temperature=0.0,
        max_tokens=100,
    )
    return response.choices[0].message.content


# ── LangGraph Node ─────────────────────────────────────────────

async def generation_node(state: PipelineState) -> PipelineState:
    """
    LangGraph generation node.
    Reads query_text + retrieved_chunks → sets generated_answer + timings.
    """
    query = state.get("query_text", "") or ""
    chunks = state.get("retrieved_chunks", [])

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Passage {i}]: {chunk.get('text', '')}")
    context = "\n".join(context_parts) if context_parts else ""

    start = time.perf_counter()

    try:
        answer = _cached_groq_generation(query, context)
    except Exception as exc:
        logger.error("generation_failed", error=str(exc))
        answer = f"Generation error: {exc}"

    elapsed_ms = (time.perf_counter() - start) * 1000

    state["generated_answer"] = answer
    state.setdefault("timings", {})["generation_ms"] = round(elapsed_ms, 2)

    logger.info(
        "generation_complete",
        answer_length=len(answer),
        elapsed_ms=round(elapsed_ms, 2),
        cached=elapsed_ms < 1.0,
    )

    return state
