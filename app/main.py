from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.config import settings
from app.services.chat_service import ChatService
from app.services.openai_service import OpenAIService, OpenAIServiceError


@asynccontextmanager
async def lifespan(app: FastAPI):
    openai_svc = OpenAIService(settings.openai_api_key, settings.model)
    app.state.chat_service = ChatService(openai_svc)
    yield


app = FastAPI(title="Cortex", lifespan=lifespan)
app.include_router(chat_router)


@app.exception_handler(OpenAIServiceError)
async def openai_error_handler(request, exc: OpenAIServiceError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})
