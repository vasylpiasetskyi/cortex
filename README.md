# Cortex

A production-ready AI backend service. Training project covering FastAPI, OpenAI Chat Completions, Redis caching, Postgres persistence, SSE streaming, function calling, structured outputs, and RAG (Retrieval-Augmented Generation) with PDF documents.

## Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| AI | OpenAI SDK (Chat Completions, Embeddings) |
| Vector DB | Qdrant (async client) |
| Cache | Redis 7 (redis-py asyncio) |
| Database | Postgres 16 + SQLAlchemy 2 async + Alembic |
| Validation | Pydantic v2 + pydantic-settings |
| PDF parsing | pypdf |
| Token counting | tiktoken |
| Logging | Loguru |
| Containers | Docker Compose |
| Package manager | uv |
| Python | 3.12+ |

## API

### Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message, get a full response |
| `POST` | `/chat/stream` | Send a message, get an SSE-streamed response |
| `POST` | `/extract` | Extract structured data (Person) from free text |

### Document Intelligence

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upload a PDF, start async indexing (202) |
| `GET` | `/documents` | List all documents for a session |
| `GET` | `/documents/{id}` | Get document status / error |
| `DELETE` | `/documents/{id}` | Delete document, chunks, and vectors |
| `POST` | `/documents/{id}/reindex` | Re-index with current embedding backend (202) |
| `POST` | `/ask` | Ask a question against indexed documents |

### POST /chat

```http
POST /chat
Content-Type: application/json

{"session_id": "abc", "message": "What is RAG?"}
```

```json
{"answer": "RAG stands for...", "session_id": "abc"}
```

### POST /chat/stream

Returns `text/event-stream`. Each token arrives as:

```
data: RAG

data:  stands

data:  for...
```

On error: `event: error\ndata: <message>`.

### POST /extract

```http
POST /extract
Content-Type: application/json

{"text": "John Smith is 35 years old"}
```

```json
{"name": "John Smith", "age": 35}
```

### POST /documents

```http
POST /documents
Content-Type: multipart/form-data

file=<pdf> session_id=abc
```

```json
{"id": 1, "filename": "contract.pdf", "status": "indexing"}
```

Indexing runs in the background. Poll `GET /documents/{id}` until `status` is `ready` or `error`.

### POST /ask

```http
POST /ask
Content-Type: application/json

{
  "session_id": "abc",
  "question": "What is the cancellation policy?",
  "document_id": 1,
  "strategy": "baseline"
}
```

```json
{
  "answer": "According to section 4.2...",
  "sources": [
    {"page": 4, "chunk_index": 12, "score": 0.91}
  ],
  "strategy": "baseline"
}
```

`document_id` is optional — omit to search all ready documents in the session.

## Document Intelligence

### Upload Pipeline

```
POST /documents
  1. Validate: PDF only, max 20 MB
  2. Save → uploads/{uuid}.pdf
  3. INSERT Document(status=indexing) → return 202
  4. BackgroundTask:
     a. pypdf   → [{page, text}]  (empty pages skipped)
     b. tiktoken cl100k_base → chunks (800 tokens, 150 overlap, step 650)
     c. INSERT chunks bulk
     d. EmbeddingService.embed_batch → vectors (OpenAI 1536-dim or local 768-dim)
     e. Qdrant upsert → collection cortex_chunks
     f. UPDATE status=ready, embedding_model=<backend model id>
     g. On error: status=error, cleanup partial Postgres rows + Qdrant vectors
```

### Retrieval Pipeline

```
POST /ask
  1. Embed question → query vector (text-embedding-3-small)
  2. Qdrant query_points (filter: session_id [+ document_id]) → top 5 chunks
  3. SELECT text FROM chunks WHERE id IN (...) → enrich results
  4. Build prompt: system = context blocks, user = question
  5. OpenAI completion → answer
  6. Return {answer, sources, strategy}
```

### Retrieval Strategies

| Strategy | Description |
|---|---|
| `baseline` | Plain Qdrant vector search, top-k results |
| `sentence_window` | Expands each match to ±2 neighboring chunks (same page) fetched from Postgres; neighbors inherit the parent match score; overlapping windows are deduplicated |
| `auto_merging` | Groups matches by `(document_id, page)`; if matched/total chunks on page ≥ 0.5, replaces them with the full page text as a single merged chunk |

## Project Structure

```
cortex/
├── app/
│   ├── api/
│   │   ├── chat.py              # /chat, /chat/stream, /extract
│   │   ├── documents.py         # /documents CRUD
│   │   └── rag.py               # /ask
│   ├── services/
│   │   ├── openai_service.py        # OpenAI wrapper (complete, stream, extract)
│   │   ├── conversation_service.py  # Redis + Postgres history
│   │   ├── chat_service.py          # Chat orchestration + tool loop
│   │   ├── document_service.py      # Upload pipeline, indexing, CRUD, reindex
│   │   ├── parser_service.py        # PDF → [{page, text}]
│   │   ├── chunking_service.py      # Pages → token chunks (800/150/650)
│   │   ├── embedding_backends.py    # EmbeddingBackend ABC, OpenAIBackend, LocalBackend
│   │   ├── embedding_service.py     # Thin wrapper over EmbeddingBackend
│   │   └── rag_service.py           # Retrieval + generation orchestration
│   ├── retrieval/
│   │   ├── base.py              # RetrieverStrategy ABC + RetrievedChunk
│   │   ├── baseline.py          # Plain vector search
│   │   ├── sentence_window.py   # Neighboring chunk expansion (WINDOW_SIZE=2)
│   │   └── auto_merging.py      # Page-level merge (MERGE_RATIO=0.5)
│   ├── tools/
│   │   ├── registry.py          # Tool schemas + dispatcher
│   │   ├── weather.py           # get_weather tool
│   │   └── calculator.py        # calculate tool (safe AST eval)
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── db.py                # SQLAlchemy Message model
│   │   └── document.py          # Document, Chunk models + DocumentStatus enum
│   ├── storage/
│   │   ├── redis.py             # Redis client factory
│   │   ├── postgres.py          # Async engine + session factory
│   │   └── qdrant.py            # AsyncQdrantClient factory
│   ├── config.py                # pydantic-settings (reads .env)
│   └── main.py                  # FastAPI app, lifespan, error handlers
├── migrations/                  # Alembic migrations
├── uploads/                     # Uploaded PDFs (gitignored)
├── tests/
│   ├── conftest.py              # Fixtures, OpenAI + Qdrant mocks
│   ├── test_chat.py
│   ├── test_stream.py
│   ├── test_conversation.py
│   ├── test_documents.py
│   ├── test_chunking.py
│   ├── test_parser.py
│   ├── test_embedding.py
│   ├── test_embedding_backends.py
│   ├── test_rag.py
│   └── test_retrieval.py
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Quick Start

### With Docker Compose

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
docker compose up --build
```

Services:

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Adminer (Postgres UI) | http://localhost:8080 |
| Redis | localhost:6379 |
| Postgres | localhost:5432 |
| Qdrant | localhost:6333 |

### Local Development

```bash
# Install dependencies
uv sync

# Start infrastructure
docker compose up redis postgres qdrant -d

# Apply migrations
uv run alembic upgrade head

# Run the app
uv run uvicorn app.main:app --reload --port 8000
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql+asyncpg://cortex:cortex@localhost:5432/cortex
QDRANT_URL=http://localhost:6333
MODEL=gpt-4o-mini
SYSTEM_PROMPT=You are a helpful assistant.
EMBEDDING_BACKEND=openai          # "openai" (default) | "local"
LOCAL_EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
```

`SYSTEM_PROMPT` is optional. `QDRANT_URL` is overridden to `http://qdrant:6333` automatically in Docker Compose.

`EMBEDDING_BACKEND=local` uses sentence-transformers on CPU (768-dim vectors). Install with `uv sync --extra local`. If you switch backends, run `make reset-vectors` and reindex existing documents — vectors from different backends are incompatible.

## Tests

OpenAI and Qdrant are fully mocked — no real API keys needed. Requires Redis and Postgres:

```bash
docker compose up redis postgres qdrant -d
uv run pytest tests/ -v
```

55 tests. OpenAI and Qdrant are mocked; Redis and Postgres run in Docker.

## Storage Architecture

### Chat: Postgres-primary + Redis-cache

Postgres is the source of truth. Redis caches the last 20 messages per session (TTL 24h).

- **Read**: Redis first → on cache miss, load from Postgres and repopulate
- **Write**: Postgres INSERT always, then Redis RPUSH + LTRIM (pipeline)

### Documents: Postgres + Qdrant

- `documents` and `chunks` tables in Postgres (source of truth for text)
- Qdrant collection `cortex_chunks`: vectors + payload `{chunk_id, document_id, session_id, page, chunk_index}`
- Text stored only in Postgres; Qdrant holds vectors and the `chunk_id` pointer
- After vector search: `SELECT text FROM chunks WHERE id IN (...)` to enrich results

## Migrations

```bash
# Generate a new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback last migration
uv run alembic downgrade -1
```
