# Voice-Enabled RAG Pipeline — HH Goa Task 2

> Voice input → Speech-to-Text → Chunking/Retrieval (Vector DB) → Answer Generation — with guardrails, latency instrumentation, and LangGraph orchestration.

## Architecture

```
┌─────────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐    ┌──────────────┐
│ Voice/Text  │───▶│ STT Node │───▶│ Off-Topic    │───▶│ Retrieval │───▶│ Generation │───▶│ Grounding    │
│   Input     │    │ (Sarvam) │    │ Guard        │    │ (Qdrant)  │    │ (Groq LLM) │    │ Guard        │
└─────────────┘    └──────────┘    └──────────────┘    └───────────┘    └────────────┘    └──────────────┘
                                        │ OFF_TOPIC          │ NO_CONTEXT                       │ NOT_GROUNDED
                                        ▼                    ▼                                  ▼
                                   ┌──────────┐        ┌──────────┐                       ┌──────────┐
                                   │  Refuse  │        │ Fallback │                       │  Refuse  │
                                   └──────────┘        └──────────┘                       └──────────┘
```

<!-- TODO: Replace with Mermaid diagram or image -->

## Tech Stack

| Component       | Technology                        |
|-----------------|-----------------------------------|
| Backend         | FastAPI (Python 3.11)             |
| Orchestration   | LangGraph                         |
| STT             | Sarvam AI API                     |
| Vector DB       | Qdrant (local, low-latency)       |
| Embeddings      | multilingual-e5-large (1024d)     |
| LLM             | Groq API (Llama 3.3 70B / 3.1 8B)|
| Dataset         | ai4bharat/MSMARCO-XI (streamed)   |

## Project Structure

```
app/
├── api/            → FastAPI routes (voice upload, text query, health, benchmark)
├── agents/         → LangGraph nodes: stt, retrieval, guardrail, generation
├── services/       → Orchestration logic, chunking strategies, ingestion, benchmark
├── schemas/        → Pydantic models for request/response, chunk metadata
├── utils/          → Latency timer decorator, audio preprocessing
├── core/           → Config, env vars, DB client init
├── exceptions/     → Custom errors + retry/error-recovery handlers
├── prompts/        → Separate .py files for generation + guardrail prompts
└── main.py         → FastAPI app entrypoint
tests/
├── test_api.py     → API endpoint tests
└── test_chunking.py→ Chunking strategy tests
```

## Chunking Strategies

1. **Fixed-Size with Overlap** — Baseline: 256 tokens, 50 token overlap
2. **Semantic Chunking** — Embedding-based split on cosine similarity drop between consecutive sentences
3. **Metadata-Aware** — Uses MSMARCO query/passage structure as natural chunk boundaries

Each chunk's metadata tracks which strategy produced it for benchmarking and analysis.

## Guardrails (LangGraph Conditional Edges)

- **Off-topic detector** — Pre-retrieval classifier via Groq (cheap, fast)
- **Grounding check** — Post-generation cosine similarity between answer and context
- **Insufficient context fallback** — Explicit refusal when top-K scores are too low
- **Retry logic** — Max 2 retries with exponential backoff on STT/LLM failures

## Quick Start

### 1. Clone & Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with Docker Compose

```bash
docker-compose up -d
```

### 3. Run locally (dev)

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 4. Ingest Data

```bash
curl -X POST "http://localhost:8000/api/v1/ingest?max_records=1000"
```

### 5. Query

```bash
# Text query
curl -X POST http://localhost:8000/api/v1/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the speed of light?"}'

# Voice query
curl -X POST http://localhost:8000/api/v1/query/voice \
  -F "file=@recording.wav" \
  -F "language=hi-IN"
```

### 6. Benchmark

```bash
curl -X POST http://localhost:8000/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"num_queries": 100, "include_generation": true}'
```

## API Endpoints

| Method | Path                    | Description                          |
|--------|-------------------------|--------------------------------------|
| GET    | `/`                     | Root — project info                  |
| GET    | `/api/v1/health`        | Health check (API + Qdrant)          |
| POST   | `/api/v1/query/voice`   | Voice RAG query (audio upload)       |
| POST   | `/api/v1/query/text`    | Text RAG query                       |
| POST   | `/api/v1/benchmark`     | Run N queries, return P50/P70/P100   |
| POST   | `/api/v1/benchmark/csv` | Benchmark results as CSV download    |
| POST   | `/api/v1/ingest`        | Trigger MSMARCO-XI ingestion         |

## Latency Target

- **Target**: Full pipeline < 200ms
- Every stage is instrumented with timestamps from request 1
- Logs flag clearly whether generation is included/excluded from the number
- `/benchmark` endpoint runs 100+ test queries and outputs P50/P70/P100

## License

MIT
