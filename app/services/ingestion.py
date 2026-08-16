"""
Ingestion service — streams MSMARCO-XI from HuggingFace and indexes into Qdrant.
Streams data (no full 55.6GB download).
"""

from __future__ import annotations

from typing import Any, Generator

import structlog

from app.core.config import get_settings
from app.core.db import get_qdrant_client, ensure_collection_exists
from app.services.chunking import chunk_all_strategies, chunk_metadata_aware
from app.agents.retrieval_node import get_embedder
from app.schemas import ChunkStrategy

logger = structlog.get_logger(__name__)

DATASET_NAME = "ai4bharat/MSMARCO-XI"


def stream_msmarco_xi(
    split: str = "train",
    language: str = "en",
    max_records: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream records from MSMARCO-XI dataset without downloading full dump.

    Yields dicts with keys like: query, positive_passages, negative_passages, etc.
    """
    from datasets import load_dataset

    logger.info(
        "streaming_dataset",
        dataset=DATASET_NAME,
        split=split,
        language=language,
    )

    ds = load_dataset(
        DATASET_NAME,
        language,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    for i, record in enumerate(ds):
        if max_records and i >= max_records:
            break
        yield record


async def ingest_to_qdrant(
    strategy: str = "metadata_aware",
    max_records: int = 1000,
    batch_size: int = 64,
    language: str = "en",
) -> dict[str, Any]:
    """
    Stream MSMARCO-XI data, chunk it, embed it, and upsert to Qdrant.

    Args:
        strategy: Which chunking strategy to use for indexing.
                  One of: "fixed_size", "semantic", "metadata_aware".
        max_records: Max records to ingest (for dev/testing).
        batch_size: Qdrant upsert batch size.
        language: MSMARCO-XI language subset.

    Returns:
        Summary dict with counts and timing.
    """
    from qdrant_client.http.models import PointStruct
    import uuid
    import time

    ensure_collection_exists()
    settings = get_settings()
    embedder = get_embedder()
    client = get_qdrant_client()

    total_chunks = 0
    total_records = 0
    start = time.perf_counter()

    batch_points: list[PointStruct] = []

    for record in stream_msmarco_xi("train", language, max_records):
        total_records += 1

        # Extract passages from MSMARCO structure
        passages = record.get("positive_passages", [])
        if not passages:
            passages = record.get("passages", [])

        query = record.get("query", "")

        for passage_data in passages:
            if isinstance(passage_data, dict):
                text = passage_data.get("text", passage_data.get("passage_text", ""))
                pid = passage_data.get("docid", passage_data.get("pid", ""))
            else:
                text = str(passage_data)
                pid = ""

            if not text.strip():
                continue

            # Chunk based on selected strategy
            chunks = chunk_metadata_aware(
                passage=text,
                query=query,
                passage_id=str(pid),
                doc_id=f"msmarco_{total_records}",
                language=language,
            )

            for chunk in chunks:
                # Embed with passage prefix for e5
                vec = embedder.encode(
                    f"passage: {chunk['text']}",
                    normalize_embeddings=True,
                ).tolist()

                payload = {
                    "text": chunk["text"],
                    **chunk["metadata"].model_dump(),
                }

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                )
                batch_points.append(point)
                total_chunks += 1

                # Batch upsert
                if len(batch_points) >= batch_size:
                    client.upsert(
                        collection_name=settings.qdrant_collection,
                        points=batch_points,
                    )
                    batch_points = []

        if total_records % 100 == 0:
            logger.info(
                "ingestion_progress",
                records=total_records,
                chunks=total_chunks,
            )

    # Flush remaining
    if batch_points:
        client.upsert(
            collection_name=settings.qdrant_collection,
            points=batch_points,
        )

    elapsed = time.perf_counter() - start

    summary = {
        "records_processed": total_records,
        "chunks_indexed": total_chunks,
        "strategy": strategy,
        "language": language,
        "elapsed_seconds": round(elapsed, 2),
        "collection": settings.qdrant_collection,
    }

    logger.info("ingestion_complete", **summary)
    return summary
