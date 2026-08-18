import json
import statistics
import time

import httpx

API_URL = "http://127.0.0.1:8000/api/v1/query/text"

# 10 test queries representative of MSMARCO data
TEST_QUERIES = [
    "calories calculator to lose weight",
    "california narcotic laws",
    "traveling physical therapist",
    "what is semantic search",
    "symptoms of acute bronchitis",
    "how to reset network settings",
    "average temperature in goa",
    "best practices for vector indexing",
    "what is speech to text transcription",
    "fastest inference engine for llm"
]

def run_benchmark(n_runs=30):
    timings = {"retrieval": [], "generation": [], "total": []}

    with httpx.Client(timeout=30.0) as client:
        # Warm-up request
        try:
            client.post(API_URL, json={"query": TEST_QUERIES[0], "language": "en"})
        except Exception:
            pass

        for i in range(n_runs):
            q = TEST_QUERIES[i % len(TEST_QUERIES)]
            start_req = time.perf_counter()
            resp = client.post(API_URL, json={"query": q, "language": "en"})
            total_elapsed = (time.perf_counter() - start_req) * 1000

            if resp.status_code == 200:
                data = resp.json()
                t_breakdown = data.get("timings", {})
                timings["retrieval"].append(t_breakdown.get("retrieval_ms", 0))
                timings["generation"].append(t_breakdown.get("generation_ms", 0))
                timings["total"].append(total_elapsed)

    def calc_percentiles(arr):
        if not arr:
            return {"p50": 9999, "p70": 9999, "p100": 9999}
        arr_sorted = sorted(arr)
        return {
            "p50": round(statistics.median(arr_sorted), 2),
            "p70": round(arr_sorted[int(len(arr_sorted) * 0.70)], 2),
            "p100": round(max(arr_sorted), 2)
        }

    report = {
        "retrieval_ms": calc_percentiles(timings["retrieval"]),
        "generation_ms": calc_percentiles(timings["generation"]),
        "total_pipeline_ms": calc_percentiles(timings["total"]),
        "samples_tested": len(timings["total"])
    }

    with open("latency_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_benchmark()
