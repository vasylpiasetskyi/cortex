from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    chat_service = request.app.state.chat_service
    answer = await chat_service.handle(body.session_id, body.message)
    return ChatResponse(answer=answer, session_id=body.session_id)


@router.post("/chat/stream")
async def stream_chat(body: ChatRequest, request: Request) -> EventSourceResponse:
    chat_service = request.app.state.chat_service

    async def generate():
        async for chunk in chat_service.handle_stream(body.session_id, body.message):
            yield {"data": chunk}

    return EventSourceResponse(generate())
