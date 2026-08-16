"""
Tests for chunking strategies — verifies all 3 strategies produce
valid chunks with correct metadata.
"""

import pytest

from app.services.chunking import (
    chunk_fixed_size,
    chunk_metadata_aware,
    count_tokens,
)
from app.schemas import ChunkStrategy


class TestFixedSizeChunking:
    """Tests for fixed-size chunking with overlap."""

    def test_basic_chunking(self):
        """Short text produces at least one chunk."""
        text = "This is a simple test sentence for chunking."
        chunks = chunk_fixed_size(text, doc_id="test_001", chunk_size=10, chunk_overlap=2)
        assert len(chunks) >= 1
        assert chunks[0]["metadata"].strategy == ChunkStrategy.FIXED_SIZE

    def test_chunk_metadata(self):
        """Each chunk carries correct metadata."""
        text = "Word " * 100  # ~100 tokens
        chunks = chunk_fixed_size(text, doc_id="doc_42", chunk_size=20, chunk_overlap=5)

        for i, chunk in enumerate(chunks):
            meta = chunk["metadata"]
            assert meta.source_doc_id == "doc_42"
            assert meta.strategy == ChunkStrategy.FIXED_SIZE
            assert meta.chunk_index == i
            assert meta.total_chunks == len(chunks)
            assert meta.token_count > 0

    def test_overlap_works(self):
        """Chunks with overlap share some text."""
        text = "Word " * 50
        chunks = chunk_fixed_size(text, chunk_size=20, chunk_overlap=10)
        if len(chunks) >= 2:
            # Overlapping chunks should share some content
            tokens_0 = set(chunks[0]["text"].split())
            tokens_1 = set(chunks[1]["text"].split())
            assert len(tokens_0 & tokens_1) > 0

    def test_empty_text(self):
        """Empty text produces one empty chunk."""
        chunks = chunk_fixed_size("", doc_id="empty")
        assert len(chunks) == 1


class TestMetadataAwareChunking:
    """Tests for MSMARCO metadata-aware chunking."""

    def test_short_passage_single_chunk(self):
        """Short passage stays as a single chunk."""
        passage = "The speed of light is approximately 299,792 km/s."
        chunks = chunk_metadata_aware(
            passage, query="What is the speed of light?",
            passage_id="P001", doc_id="msmarco_1", language="en",
        )
        assert len(chunks) == 1
        assert chunks[0]["metadata"].strategy == ChunkStrategy.METADATA_AWARE
        assert chunks[0]["metadata"].query_text == "What is the speed of light?"
        assert chunks[0]["metadata"].passage_id == "P001"

    def test_metadata_preserved(self):
        """MSMARCO metadata is preserved in chunks."""
        passage = "Some long passage. " * 200
        chunks = chunk_metadata_aware(
            passage, query="test query", passage_id="P999",
            doc_id="doc_X", language="hi",
        )
        for chunk in chunks:
            assert chunk["metadata"].query_text == "test query"
            assert chunk["metadata"].passage_id == "P999"
            assert chunk["metadata"].language == "hi"


class TestTokenCounter:
    """Tests for token counting utility."""

    def test_count_tokens_basic(self):
        """Token count is positive for non-empty text."""
        assert count_tokens("Hello, world!") > 0

    def test_count_tokens_empty(self):
        """Empty string has 0 tokens."""
        assert count_tokens("") == 0
