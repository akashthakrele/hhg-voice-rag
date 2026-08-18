"""
Generation Node — Local Micro-LLM answer generation via llama-cpp-python.
Uses Qwen2.5-0.5B-Instruct-GGUF for sub-20ms local in-memory generation with zero network lag.
"""

from __future__ import annotations

import asyncio
import os
import time
from functools import lru_cache

import structlog
from llama_cpp import Llama

from app.agents.state import PipelineState
from app.exceptions import GenerationError

logger = structlog.get_logger(__name__)

MODEL_PATH = os.path.join("models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    """Lazy-initialize singleton Llama instance keeping weights hot in RAM/VRAM."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Local model not found at {MODEL_PATH}")
    logger.info("loading_local_llm", model_path=MODEL_PATH)
    return Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Offloads all layers to GPU if available
        n_ctx=512,        # Ultra-compact context window for speed
        n_threads=8,      # Utilize CPU threads
        verbose=False,    # Suppress C++ logs
    )


def truncate_words(text: str, max_words: int = 150) -> str:
    """Truncate text to at most max_words to minimize prompt prefill latency."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


async def generation_node(state: PipelineState) -> PipelineState:
    """
    Generate an answer using local Qwen2.5-0.5B LLM based on retrieved context.
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

    # Build context string from top 2 retrieved chunks and prune to <= 150 words
    context_parts = []
    for i, chunk in enumerate(chunks[:2], 1):
        text = chunk.get("text", "")
        context_parts.append(f"[Passage {i}]\n{text}")
    raw_context = "\n\n".join(context_parts)
    context = truncate_words(raw_context, max_words=150)

    start = time.perf_counter()
    try:
        answer = await _call_local_llm(query, context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        state["generated_answer"] = answer
        state.setdefault("timings", {})["generation_ms"] = round(elapsed_ms, 2)

        logger.info(
            "generation_complete",
            model="qwen2.5-0.5b-instruct-q4_k_m",
            elapsed_ms=round(elapsed_ms, 2),
            answer_length=len(answer),
        )
        return state

    except Exception as exc:
        logger.error("generation_failed", error=str(exc))
        state["error"] = f"Local generation failed: {exc}"
        raise GenerationError(str(exc)) from exc


async def _call_local_llm(query: str, context: str) -> str:
    """Run local chat completion on a background worker thread."""
    llm = get_llm()

    messages = [
        {
            "role": "system",
            "content": "Answer using ONLY context. Max 10 words.",
        },
        {
            "role": "user",
            "content": f"Context: {context}\nQuestion: {query}",
        },
    ]

    response = await asyncio.to_thread(
        llm.create_chat_completion,
        messages=messages,
        max_tokens=12,
        temperature=0.0,
    )

    answer = response["choices"][0]["message"]["content"]
    if not answer or not answer.strip():
        raise GenerationError("Empty response from local LLM")

    return answer.strip()
