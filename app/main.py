import json
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.chat import router as chat_router
from app.config import settings
from app.context import get_request_id
from app.middleware.request_id import RequestIDMiddleware
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.storage.postgres import get_engine, get_session_factory, init_db
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
async def lifespan(app: FastAPI):  # noqa: F811
    setup_logging()
    redis = get_redis(settings.redis_url)
    engine = get_engine(settings.database_url)
    await init_db(engine)
    session_factory = get_session_factory(engine)

    openai_svc = OpenAIService(settings.openai_api_key, settings.model)
    conv_svc = ConversationService(redis, session_factory)
    app.state.chat_service = ChatService(openai_svc, conv_svc, settings.system_prompt)

    yield

    await redis.aclose()
    await engine.dispose()


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


@app.exception_handler(OpenAIServiceError)
async def openai_error_handler(request, exc: OpenAIServiceError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run("app.main:app", reload=True, port=8000)
    uvicorn.run("app.main:app", reload=False, port=8000)
