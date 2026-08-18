"""
Benchmark service — runs N test queries and computes P50/P70/P100 latency stats.
Outputs results as CSV for the /benchmark endpoint.
"""

from __future__ import annotations

import csv
import io

import numpy as np
import structlog

from app.schemas import BenchmarkResult, PipelineTimings
from app.services.pipeline import run_text_pipeline

logger = structlog.get_logger(__name__)

# Sample queries for benchmarking (subset — extend for real runs)
BENCHMARK_QUERIES = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What are the symptoms of diabetes?",
    "Who invented the telephone?",
    "What is machine learning?",
    "How far is the moon from Earth?",
    "What causes earthquakes?",
    "What is the speed of light?",
    "How does a vaccine work?",
    "What is the largest ocean on Earth?",
    "What is DNA?",
    "How do solar panels work?",
    "What is the population of India?",
    "What are black holes?",
    "How is paper made?",
    "What is the greenhouse effect?",
    "What causes rain?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "How do computers store data?",
]


async def run_benchmark(
    num_queries: int = 100,
    include_generation: bool = True,
) -> BenchmarkResult:
    """
    Run benchmark across N queries and compute latency percentiles.

    Cycles through BENCHMARK_QUERIES if num_queries > len(queries).
    """
    all_timings: list[PipelineTimings] = []
    total_times: list[float] = []

    for i in range(num_queries):
        query = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]

        try:
            response = await run_text_pipeline(query)
            timing = response.timings

            # If generation excluded, subtract it from total
            if not include_generation and timing.generation_ms:
                adjusted = timing.total_ms - timing.generation_ms
                timing = timing.model_copy(update={"total_ms": adjusted})

            all_timings.append(timing)
            total_times.append(timing.total_ms)

        except Exception as exc:
            logger.warning("benchmark_query_failed", query=query, error=str(exc))
            continue

        if (i + 1) % 10 == 0:
            logger.info("benchmark_progress", completed=i + 1, total=num_queries)

    if not total_times:
        return BenchmarkResult(
            num_queries=num_queries,
            include_generation=include_generation,
            p50_ms=0.0,
            p70_ms=0.0,
            p100_ms=0.0,
            mean_ms=0.0,
        )

    arr = np.array(total_times)
    result = BenchmarkResult(
        num_queries=len(total_times),
        include_generation=include_generation,
        p50_ms=round(float(np.percentile(arr, 50)), 2),
        p70_ms=round(float(np.percentile(arr, 70)), 2),
        p100_ms=round(float(np.max(arr)), 2),
        mean_ms=round(float(np.mean(arr)), 2),
        all_timings=all_timings,
    )

    logger.info(
        "benchmark_complete",
        num_queries=result.num_queries,
        p50=result.p50_ms,
        p70=result.p70_ms,
        p100=result.p100_ms,
        mean=result.mean_ms,
        includes_generation=include_generation,
    )

    return result


def timings_to_csv(result: BenchmarkResult) -> str:
    """Convert benchmark results to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "query_index", "total_ms", "stt_ms", "retrieval_ms",
        "generation_ms", "guardrail_ms", "exceeds_target",
    ])

    for i, t in enumerate(result.all_timings):
        writer.writerow([
            i + 1,
            t.total_ms,
            t.stt_ms or "",
            t.retrieval_ms or "",
            t.generation_ms or "",
            t.guardrail_ms or "",
            t.exceeds_target,
        ])

    # Summary row
    writer.writerow([])
    writer.writerow(["metric", "value"])
    writer.writerow(["P50_ms", result.p50_ms])
    writer.writerow(["P70_ms", result.p70_ms])
    writer.writerow(["P100_ms", result.p100_ms])
    writer.writerow(["Mean_ms", result.mean_ms])
    writer.writerow(["Num_Queries", result.num_queries])
    writer.writerow(["Includes_Generation", result.include_generation])

    return output.getvalue()
