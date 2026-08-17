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
├── frontend/           → Zero-build production-ready UI (plain HTML/CSS/JS)
│   ├── index.html      → Main UI: voice recorder, text input, latency badges, grounding indicator
│   ├── style.css       → Dark-mode responsive styling & micro-interactions
│   ├── app.js          → MediaRecorder voice capture, API client, error handling
│   └── assets/         → Static SVG icons and branding assets
├── app/                → Backend Python application (FastAPI & LangGraph)
│   ├── api/            → FastAPI routes (voice upload, text query, health, benchmark)
│   ├── agents/         → LangGraph nodes: stt, retrieval, guardrail, generation
│   ├── services/       → Orchestration logic, chunking strategies, ingestion, benchmark
│   ├── schemas/        → Pydantic models for request/response, chunk metadata
│   ├── utils/          → Latency timer decorator, audio preprocessing
│   ├── core/           → Config, env vars, DB client init
│   ├── exceptions/     → Custom errors + retry/error-recovery handlers
│   ├── prompts/        → Separate .py files for generation + guardrail prompts
│   └── main.py         → FastAPI app entrypoint & frontend static files mount
└── tests/
    ├── test_api.py     → API endpoint & static frontend tests
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

## Local Development

### 1. Environment Configuration

Copy the example environment configuration file and supply your API keys:

```bash
cp .env.example .env
```

Key environment variables configured in `.env.example`:
- `GROQ_API_KEY` — Fast LLM inference & guardrail classification
- `SARVAM_API_KEY` — Indian language Speech-to-Text (STT)
- `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_URL` — Vector database connection
- `LATENCY_TARGET_MS` — Pipeline latency threshold (default: `200`)

### 2. Python Environment & Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux / macOS

# Install required dependencies
pip install -r requirements.txt
```

### 3. Run Backend & Frontend

Start the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Accessing the Application

- **Frontend UI**: Open [http://localhost:8000/](http://localhost:8000/) directly in your web browser.
- **Interactive API Docs (Swagger)**: Visit [http://localhost:8000/docs](http://localhost:8000/docs).
- **Alternative Docs (ReDoc)**: Visit [http://localhost:8000/redoc](http://localhost:8000/redoc).
- **Health Check**: Visit [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health).

## Testing & Ingestion

### Ingest Data

```bash
curl -X POST "http://localhost:8000/api/v1/ingest?max_records=1000"
```

### Query Examples

```bash
# Text query
curl -X POST http://localhost:8000/api/v1/query/text \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the speed of light?", "language": "en"}'

# Voice query
curl -X POST http://localhost:8000/api/v1/query/voice \
  -F "file=@recording.wav" \
  -F "language=hi-IN"
```

### Run Benchmarks

```bash
curl -X POST http://localhost:8000/api/v1/benchmark \
  -H "Content-Type: application/json" \
  -d '{"num_queries": 100, "include_generation": true}'
```

## API Endpoints

| Method | Path                    | Description                                  |
|--------|-------------------------|----------------------------------------------|
| GET    | `/`                     | Frontend Web UI (HTML)                       |
| GET    | `/docs`                 | Swagger Interactive API Documentation        |
| GET    | `/api/v1/health`        | Health check (API + Qdrant status)           |
| POST   | `/api/v1/query/voice`   | Voice RAG query (audio upload → STT → RAG)   |
| POST   | `/api/v1/query/text`    | Text RAG query (Direct retrieval + LLM)      |
| POST   | `/api/v1/benchmark`     | Run N queries, return P50/P70/P100 latency   |
| POST   | `/api/v1/benchmark/csv` | Benchmark results as downloadable CSV        |
| POST   | `/api/v1/ingest`        | Trigger background MSMARCO-XI ingestion      |

## Latency Target

- **Target**: Full pipeline < 200ms
- Every stage is instrumented with timestamps from request 1
- Logs flag clearly whether generation is included/excluded from the number
- `/benchmark` endpoint runs 100+ test queries and outputs P50/P70/P100

## License

MIT
