# Cortex

A production-ready AI chat backend service. Training project covering FastAPI, OpenAI Chat Completions, Redis caching, Postgres persistence, SSE streaming, function calling, and structured outputs.

## Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| AI | OpenAI SDK (Chat Completions) |
| Cache | Redis 7 (redis-py asyncio) |
| Database | Postgres 16 + SQLAlchemy 2 async + Alembic |
| Validation | Pydantic v2 + pydantic-settings |
| Logging | Loguru |
| Containers | Docker Compose |
| Package manager | uv |
| Python | 3.12+ |

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message, get a full response |
| `POST` | `/chat/stream` | Send a message, get an SSE-streamed response |
| `POST` | `/extract` | Extract structured data (Person) from free text |

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

## Project Structure

```
cortex/
├── app/
│   ├── api/chat.py              # Route handlers
│   ├── services/
│   │   ├── openai_service.py    # OpenAI wrapper (complete, stream, extract)
│   │   ├── conversation_service.py  # Redis + Postgres history
│   │   └── chat_service.py      # Orchestration + tool loop
│   ├── tools/
│   │   ├── registry.py          # Tool schemas + dispatcher
│   │   └── weather.py           # Demo get_weather tool
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── db.py                # SQLAlchemy Message model
│   ├── storage/
│   │   ├── redis.py             # Redis client factory
│   │   └── postgres.py          # Async engine + session factory
│   ├── config.py                # pydantic-settings (reads .env)
│   └── main.py                  # FastAPI app, lifespan, error handlers
├── migrations/                  # Alembic migrations
├── tests/
│   ├── conftest.py              # Fixtures, OpenAI mock
│   ├── test_chat.py
│   ├── test_stream.py
│   └── test_conversation.py
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
| Adminer (Postgres UI) | http://localhost:8080 |
| Redis | localhost:6379 |
| Postgres | localhost:5432 |

### Local Development

```bash
# Install dependencies
uv sync

# Start infrastructure
docker compose up redis postgres -d

# Run the app
uv run uvicorn app.main:app --reload
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
OPENAI_API_KEY=sk-...
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql+asyncpg://cortex:cortex@localhost:5432/cortex
MODEL=gpt-4o-mini
```

## Tests

Tests run against real in-memory SQLite + Redis via Docker. OpenAI is fully mocked.

```bash
uv run pytest tests/ -v
```

All 9 tests run without a real OpenAI API key.

## Storage Architecture

Postgres is the source of truth. Redis caches the last 20 messages per session (TTL 24h).

- **Read**: Redis first → on cache miss, load from Postgres and repopulate Redis
- **Write**: Postgres INSERT always, then Redis RPUSH + LTRIM (pipeline)

## Function Calling

`ChatService.handle()` runs a tool loop:

1. Send history + user message to OpenAI with tool schemas
2. If `finish_reason == "tool_calls"`: execute the tool, append result, call OpenAI again
3. Repeat until a plain text response arrives

Current tools: `get_weather(city)` — returns hardcoded demo weather.

## Migrations

```bash
# Generate a new migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head
```
