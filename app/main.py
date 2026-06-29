import json
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from qdrant_client.http.exceptions import UnexpectedResponse as QdrantError

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.rag import router as rag_router
from app.config import settings
from app.context import get_request_id
from app.middleware.request_id import RequestIDMiddleware
from app.services.chat_service import ChatService
from app.services.chunking_service import ChunkingService
from app.services.conversation_service import ConversationService
from app.services.document_service import DocumentService
from app.services.embedding_backends import LocalBackend, OpenAIBackend
from app.services.embedding_service import EmbeddingService
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.services.parser_service import ParserService
from app.services.rag_service import RAGService
from app.storage.postgres import get_engine, get_session_factory, init_db
from app.storage.qdrant import get_qdrant_client, init_collection
from app.storage.redis import get_redis


def _json_formatter(record: dict) -> str:
    entry = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "msg": record["message"],
        "request_id": get_request_id(),
    }
    entry.update(record["extra"])
    record["extra"]["_json"] = json.dumps(entry, default=str)
    return "{extra[_json]}\n"


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, format=_json_formatter)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    openai_svc = OpenAIService(settings.openai_api_key, settings.model)

    if settings.embedding_backend == "local":
        backend = LocalBackend(settings.local_embedding_model)
    else:
        backend = OpenAIBackend(openai_svc.client)
    embedding_svc = EmbeddingService(backend)

    redis = get_redis(settings.redis_url)
    engine = get_engine(settings.database_url)
    await init_db(engine)
    session_factory = get_session_factory(engine)

    qdrant = get_qdrant_client(settings.qdrant_url)
    await init_collection(qdrant, embedding_svc.vector_size)

    conv_svc = ConversationService(redis, session_factory)
    app.state.chat_service = ChatService(openai_svc, conv_svc, settings.system_prompt)
    app.state.qdrant = qdrant
    app.state.session_factory = session_factory
    app.state.openai_client = openai_svc.client

    parser = ParserService()
    chunker = ChunkingService()
    doc_svc = DocumentService(session_factory, parser, chunker, embedding_svc, qdrant)

    app.state.doc_service = doc_svc
    app.state.embedding_svc = embedding_svc

    rag_svc = RAGService(openai_svc, embedding_svc, doc_svc, session_factory, qdrant)
    app.state.rag_service = rag_svc

    yield

    await redis.aclose()
    await engine.dispose()
    await qdrant.close()


app = FastAPI(title="Cortex", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(RequestIDMiddleware)

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(rag_router)


@app.exception_handler(OpenAIServiceError)
async def openai_error_handler(request, exc: OpenAIServiceError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(QdrantError)
async def qdrant_error_handler(request, exc: QdrantError):
    return JSONResponse(status_code=503, content={"detail": f"Vector store unavailable: {exc}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", reload=False, port=8000)
