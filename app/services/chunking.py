"""
Chunking service — implements 3 strategies for MSMARCO-XI dataset.
Each chunk carries metadata tracking which strategy produced it.
"""

from __future__ import annotations

from typing import Any

import structlog
import tiktoken

from app.core.config import get_settings
from app.schemas import ChunkMetadata, ChunkStrategy

logger = structlog.get_logger(__name__)

# Lazy-loaded tokenizer
_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_get_tokenizer().encode(text))


# ═══════════════════════════════════════════════════════════
# Strategy 1: Fixed-Size with Overlap
# ═══════════════════════════════════════════════════════════


def chunk_fixed_size(
    text: str,
    doc_id: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Split text into fixed-size token chunks with overlap.

    Args:
        text: Input text to chunk.
        doc_id: Source document identifier.
        chunk_size: Tokens per chunk (default from settings).
        chunk_overlap: Overlap tokens between chunks (default from settings).

    Returns:
        List of {"text": ..., "metadata": ChunkMetadata}.
    """
    settings = get_settings()
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    tokenizer = _get_tokenizer()

    tokens = tokenizer.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        metadata = ChunkMetadata(
            source_doc_id=doc_id,
            strategy=ChunkStrategy.FIXED_SIZE,
            chunk_index=len(chunks),
            token_count=len(chunk_tokens),
        )

        chunks.append({"text": chunk_text, "metadata": metadata})

        if end >= len(tokens):
            break
        start += size - overlap

    # Update total_chunks
    for c in chunks:
        c["metadata"].total_chunks = len(chunks)

    return chunks


# ═══════════════════════════════════════════════════════════
# Strategy 2: Semantic Chunking
# ═══════════════════════════════════════════════════════════


def chunk_semantic(
    text: str,
    doc_id: str = "",
    similarity_threshold: float = 0.5,
    min_chunk_tokens: int = 50,
    max_chunk_tokens: int = 512,
) -> list[dict[str, Any]]:
    """
    Split text based on embedding similarity drops between sentences.

    Process:
    1. Split into sentences
    2. Compute pairwise cosine similarity of consecutive sentences
    3. Split at points where similarity drops below threshold
    4. Merge tiny chunks

    Args:
        text: Input text.
        doc_id: Source document identifier.
        similarity_threshold: Break when similarity drops below this.
        min_chunk_tokens: Minimum tokens per chunk (merge if smaller).
        max_chunk_tokens: Maximum tokens per chunk (split if larger).
    """
    import nltk
    import numpy as np

    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 1:
        metadata = ChunkMetadata(
            source_doc_id=doc_id,
            strategy=ChunkStrategy.SEMANTIC,
            chunk_index=0,
            total_chunks=1,
            token_count=count_tokens(text),
        )
        return [{"text": text, "metadata": metadata}]

    # Get embeddings for each sentence
    from app.agents.retrieval_node import get_embedder

    embedder = get_embedder()
    prefixed = [f"passage: {s}" for s in sentences]
    embeddings = embedder.encode(prefixed, normalize_embeddings=True)

    # Find break points based on similarity drops
    break_points = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        if sim < similarity_threshold:
            break_points.append(i + 1)

    # Create chunks from break points
    chunks = []
    prev = 0
    for bp in break_points:
        chunk_text = " ".join(sentences[prev:bp])
        tokens = count_tokens(chunk_text)

        # Enforce max size — split further if needed
        if tokens > max_chunk_tokens:
            sub_chunks = chunk_fixed_size(chunk_text, doc_id, max_chunk_tokens, 50)
            for sc in sub_chunks:
                sc["metadata"].strategy = ChunkStrategy.SEMANTIC
            chunks.extend(sub_chunks)
        elif tokens >= min_chunk_tokens:
            metadata = ChunkMetadata(
                source_doc_id=doc_id,
                strategy=ChunkStrategy.SEMANTIC,
                chunk_index=len(chunks),
                token_count=tokens,
            )
            chunks.append({"text": chunk_text, "metadata": metadata})
        prev = bp

    # Last chunk
    if prev < len(sentences):
        chunk_text = " ".join(sentences[prev:])
        tokens = count_tokens(chunk_text)
        if tokens >= min_chunk_tokens or not chunks:
            metadata = ChunkMetadata(
                source_doc_id=doc_id,
                strategy=ChunkStrategy.SEMANTIC,
                chunk_index=len(chunks),
                token_count=tokens,
            )
            chunks.append({"text": chunk_text, "metadata": metadata})
        elif chunks:
            # Merge tiny tail into last chunk
            chunks[-1]["text"] += " " + chunk_text
            chunks[-1]["metadata"].token_count += tokens

    # Update total_chunks and indices
    for i, c in enumerate(chunks):
        c["metadata"].chunk_index = i
        c["metadata"].total_chunks = len(chunks)

    return chunks


# ═══════════════════════════════════════════════════════════
# Strategy 3: Metadata-Aware (MSMARCO structure)
# ═══════════════════════════════════════════════════════════


def chunk_metadata_aware(
    passage: str,
    query: str = "",
    passage_id: str = "",
    doc_id: str = "",
    language: str = "en",
) -> list[dict[str, Any]]:
    """
    Use MSMARCO query/passage structure as natural chunk boundaries.

    Each MSMARCO passage is already a semantically coherent unit paired
    with a query. We keep this structure intact and enrich metadata.

    For long passages, we sub-chunk with fixed-size but preserve the
    query and passage_id in metadata.
    """
    max_tokens = 512
    tokens = count_tokens(passage)

    if tokens <= max_tokens:
        # Keep passage as a single chunk — it's already a semantic unit
        metadata = ChunkMetadata(
            source_doc_id=doc_id,
            strategy=ChunkStrategy.METADATA_AWARE,
            chunk_index=0,
            total_chunks=1,
            token_count=tokens,
            query_text=query,
            passage_id=passage_id,
            language=language,
        )
        return [{"text": passage, "metadata": metadata}]

    # Sub-chunk long passages but preserve MSMARCO metadata
    sub_chunks = chunk_fixed_size(passage, doc_id, max_tokens, 50)
    for i, sc in enumerate(sub_chunks):
        sc["metadata"].strategy = ChunkStrategy.METADATA_AWARE
        sc["metadata"].query_text = query
        sc["metadata"].passage_id = passage_id
        sc["metadata"].language = language
        sc["metadata"].chunk_index = i
        sc["metadata"].total_chunks = len(sub_chunks)

    return sub_chunks


# ═══════════════════════════════════════════════════════════
# Orchestrator — run all 3 strategies
# ═══════════════════════════════════════════════════════════


def chunk_all_strategies(
    text: str,
    doc_id: str = "",
    query: str = "",
    passage_id: str = "",
    language: str = "en",
) -> dict[str, list[dict[str, Any]]]:
    """
    Run all 3 chunking strategies on the same text.

    Returns:
        Dict mapping strategy name → list of chunks.
        Each chunk has {"text": ..., "metadata": ChunkMetadata}.
    """
    return {
        "fixed_size": chunk_fixed_size(text, doc_id),
        "semantic": chunk_semantic(text, doc_id),
        "metadata_aware": chunk_metadata_aware(
            text, query=query, passage_id=passage_id,
            doc_id=doc_id, language=language,
        ),
    }
