"""
Ingestion service — streams MSMARCO-XI from HuggingFace and indexes into Qdrant.
Streams data (no full 55.6GB download).
"""

from __future__ import annotations

from typing import Any, Generator

import structlog

from app.core.config import get_settings
from app.core.db import get_qdrant_client, ensure_collection_exists, recreate_collection
from app.services.chunking import chunk_all_strategies, chunk_metadata_aware
from app.agents.retrieval_node import get_embedder
from app.schemas import ChunkStrategy

logger = structlog.get_logger(__name__)

DATASET_NAME = "ai4bharat/MSMARCO-XI"

LANGUAGE_FILE_MAP: dict[str, str] = {
    "hi": "train/hintrain.parquet",
    "hin": "train/hintrain.parquet",
    "hi-in": "train/hintrain.parquet",
    "en": "train/hintrain.parquet",
    "eng": "train/hintrain.parquet",
    "en-in": "train/hintrain.parquet",
    "en-us": "train/hintrain.parquet",
    "eng_latn": "train/hintrain.parquet",
    "mr": "train/martrain.parquet",
    "mar": "train/martrain.parquet",
    "mr-in": "train/martrain.parquet",
    "bn": "train/bentrain.parquet",
    "ben": "train/bentrain.parquet",
    "bn-in": "train/bentrain.parquet",
    "ta": "train/tamtrain.parquet",
    "tam": "train/tamtrain.parquet",
    "ta-in": "train/tamtrain.parquet",
    "te": "train/telval.parquet",
    "tel": "train/telval.parquet",
    "te-in": "train/telval.parquet",
    "gu": "train/gujtrain.parquet",
    "guj": "train/gujtrain.parquet",
    "gu-in": "train/gujtrain.parquet",
    "kn": "train/kantrain.parquet",
    "kan": "train/kantrain.parquet",
    "kn-in": "train/kantrain.parquet",
    "ml": "train/maltrain.parquet",
    "mal": "train/maltrain.parquet",
    "ml-in": "train/maltrain.parquet",
    "ur": "train/urdtrain.parquet",
    "urd": "train/urdtrain.parquet",
    "ur-in": "train/urdtrain.parquet",
    "pa": "train/pantrain.parquet",
    "pan": "train/pantrain.parquet",
    "pa-in": "train/pantrain.parquet",
    "or": "train/oritrain.parquet",
    "ori": "train/oritrain.parquet",
    "or-in": "train/oritrain.parquet",
    "ne": "train/neptrain.parquet",
    "nep": "train/neptrain.parquet",
    "as": "train/asmtrain.parquet",
    "asm": "train/asmtrain.parquet",
    "sa": "train/santrain.parquet",
    "san": "train/santrain.parquet",
}


def stream_msmarco_xi(
    split: str = "train",
    language: str = "en",
    max_records: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream records from MSMARCO-XI dataset without downloading full dump.

    Yields dicts with keys: query, Eng_Query, passages, target_lang, source_lang, etc.
    """
    import os
    from datasets import load_dataset

    logger.info(
        "streaming_dataset",
        dataset=DATASET_NAME,
        split=split,
        language=language,
    )

    raw_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or ""
    token = raw_token.strip() if raw_token.strip() else None
    lang_clean = (language or "").strip().lower()

    # 1. Fast path: load via cached ParquetFile using huggingface_hub
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        import pyarrow.parquet as pq

        target_file = LANGUAGE_FILE_MAP.get(
            lang_clean,
            f"{split}/hintrain.parquet" if split == "train" else f"{split}/hinval.parquet",
        )
        val_file = target_file.replace("train/", "validation/").replace("train.parquet", "val.parquet")

        # Check local cache first without network call
        cached = (
            try_to_load_from_cache(DATASET_NAME, target_file, repo_type="dataset")
            or try_to_load_from_cache(DATASET_NAME, val_file, repo_type="dataset")
        )

        if not cached:
            # Prefer validation file for fast download when batch is <= 10000
            file_to_download = val_file if (max_records and max_records <= 10000) or split == "validation" else target_file
            try:
                cached = hf_hub_download(
                    repo_id=DATASET_NAME,
                    filename=file_to_download,
                    repo_type="dataset",
                    token=token,
                )
            except Exception:
                cached = hf_hub_download(
                    repo_id=DATASET_NAME,
                    filename=val_file,
                    repo_type="dataset",
                    token=token,
                )

        local_parquet = cached
        pf = pq.ParquetFile(local_parquet)
        yielded = 0
        for batch in pf.iter_batches(batch_size=min(max_records or 64, 64)):
            records = batch.to_pylist()
            for record in records:
                if lang_clean and lang_clean not in ("all", "any", "*"):
                    target_lang = str(record.get("target_lang") or "").lower()
                    if lang_clean in ("en", "eng", "en-in", "en-us", "eng_latn"):
                        pass  # All records contain English passages
                    else:
                        lang_code = lang_clean.split("-")[0]
                        is_match = (
                            lang_clean in target_lang
                            or lang_code in target_lang
                            or (lang_code == "hi" and "hin" in target_lang)
                            or (lang_code == "mr" and "mar" in target_lang)
                            or (lang_code == "bn" and "ben" in target_lang)
                            or (lang_code == "ta" and "tam" in target_lang)
                            or (lang_code == "te" and "tel" in target_lang)
                            or (lang_code == "gu" and "guj" in target_lang)
                            or (lang_code == "kn" and "kan" in target_lang)
                            or (lang_code == "ml" and "mal" in target_lang)
                            or (lang_code == "ur" and "urd" in target_lang)
                            or (lang_code == "pa" and "pan" in target_lang)
                            or (lang_code == "or" and "ori" in target_lang)
                            or (lang_code == "ne" and "nep" in target_lang)
                            or (lang_code == "as" and "asm" in target_lang)
                            or (lang_code == "sa" and "san" in target_lang)
                        )
                        if not is_match:
                            continue

                yield record
                yielded += 1
                if max_records and yielded >= max_records:
                    return
        return

    except Exception as exc:
        logger.warning("hf_hub_read_fallback", error=str(exc))

    # 2. Fallback: streaming via datasets.load_dataset
    if lang_clean in LANGUAGE_FILE_MAP:
        data_files = {split: LANGUAGE_FILE_MAP[lang_clean]}
        ds = load_dataset(
            DATASET_NAME,
            data_files=data_files,
            split=split,
            streaming=True,
            trust_remote_code=True,
            token=token,
        )
    else:
        ds = load_dataset(
            DATASET_NAME,
            split=split,
            streaming=True,
            trust_remote_code=True,
            token=token,
        )

    yielded = 0

    for record in ds:
        # Post-loading language filtering if specified
        if lang_clean and lang_clean not in ("all", "any", "*"):
            target_lang = str(record.get("target_lang") or "").lower()
            source_lang = str(record.get("source_lang") or "").lower()

            if lang_clean in ("en", "eng", "en-in", "en-us", "eng_latn"):
                pass  # All MSMARCO-XI records contain English passages
            else:
                lang_code = lang_clean.split("-")[0]
                is_match = (
                    lang_clean in target_lang
                    or lang_code in target_lang
                    or (lang_code == "hi" and "hin" in target_lang)
                    or (lang_code == "mr" and "mar" in target_lang)
                    or (lang_code == "bn" and "ben" in target_lang)
                    or (lang_code == "ta" and "tam" in target_lang)
                    or (lang_code == "te" and "tel" in target_lang)
                    or (lang_code == "gu" and "guj" in target_lang)
                    or (lang_code == "kn" and "kan" in target_lang)
                    or (lang_code == "ml" and "mal" in target_lang)
                    or (lang_code == "ur" and "urd" in target_lang)
                    or (lang_code == "pa" and "pan" in target_lang)
                    or (lang_code == "or" and "ori" in target_lang)
                    or (lang_code == "ne" and "nep" in target_lang)
                    or (lang_code == "as" and "asm" in target_lang)
                    or (lang_code == "sa" and "san" in target_lang)
                )
                if not is_match:
                    continue

        yield record
        yielded += 1
        if max_records and yielded >= max_records:
            break


async def ingest_to_qdrant(
    strategy: str = "metadata_aware",
    max_records: int = 1000,
    batch_size: int = 64,
    language: str = "en",
    clear_existing: bool = False,
) -> dict[str, Any]:
    """
    Stream MSMARCO-XI data, chunk it, embed it, and upsert to Qdrant.

    Args:
        strategy: Which chunking strategy to use for indexing.
                  One of: "fixed_size", "semantic", "metadata_aware".
        max_records: Max records to ingest (for dev/testing).
        batch_size: Qdrant upsert batch size.
        language: MSMARCO-XI language subset.
        clear_existing: If True, delete and recreate collection before indexing.

    Returns:
        Summary dict with counts and timing.
    """
    from qdrant_client.http.models import PointStruct
    import uuid
    import time

    if clear_existing:
        recreate_collection()
    else:
        ensure_collection_exists()

    settings = get_settings()
    embedder = get_embedder()
    client = get_qdrant_client()

    total_chunks = 0
    total_records = 0
    start = time.perf_counter()

    batch_points: list[PointStruct] = []
    lang_lower = (language or "en").lower()
    is_english = lang_lower in ("en", "eng", "en-in", "en-us", "eng_latn")

    for record in stream_msmarco_xi("train", language, max_records):
        total_records += 1

        # Extract passages and query from MSMARCO-XI structure
        passages_raw = record.get("passages") or record.get("positive_passages", [])
        candidate_passages: list[tuple[str, str]] = []  # (text, passage_id)
        query = ""

        if isinstance(passages_raw, dict):
            # MSMARCO-XI schema: {'English_passages': [...], 'Translated_passages': [...], 'is_selected': [...]}
            eng_passages = passages_raw.get("English_passages", [])
            trans_passages = passages_raw.get("Translated_passages", [])
            is_selected = passages_raw.get("is_selected", [])

            if is_english:
                query = record.get("Eng_Query") or record.get("query", "")
                selected_texts = eng_passages if eng_passages else trans_passages
            else:
                query = record.get("query") or record.get("Eng_Query", "")
                selected_texts = trans_passages if trans_passages else eng_passages

            query_id = str(record.get("query_id", total_records))
            has_selected = any(sel == 1 for sel in is_selected) if is_selected else False

            for idx, p_text in enumerate(selected_texts):
                if not p_text or not str(p_text).strip():
                    continue
                if has_selected and idx < len(is_selected) and is_selected[idx] != 1:
                    continue
                candidate_passages.append((str(p_text).strip(), f"{query_id}_{idx}"))

        elif isinstance(passages_raw, list):
            query = record.get("query", "") or record.get("Eng_Query", "")
            for idx, p_item in enumerate(passages_raw):
                if isinstance(p_item, dict):
                    p_text = p_item.get("text", p_item.get("passage_text", ""))
                    pid = p_item.get("docid", p_item.get("pid", f"{total_records}_{idx}"))
                else:
                    p_text = str(p_item)
                    pid = f"{total_records}_{idx}"
                if p_text.strip():
                    candidate_passages.append((p_text.strip(), str(pid)))

        if not query:
            query = record.get("query") or record.get("Eng_Query") or ""

        for text, pid in candidate_passages:
            # Chunk based on selected strategy
            chunks = chunk_metadata_aware(
                passage=text,
                query=query,
                passage_id=str(pid),
                doc_id=f"msmarco_{total_records}",
                language=language,
            )

            for chunk in chunks:
                # Deterministic chunk & point ID based on strategy, language, passage_id, chunk_index, and text
                p_id = chunk["metadata"].passage_id or str(pid)
                c_idx = chunk["metadata"].chunk_index
                chunk_unique_str = f"{strategy}:{language}:{p_id}:{c_idx}:{chunk['text']}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_unique_str))
                chunk["metadata"].chunk_id = point_id

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
                    id=point_id,
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
