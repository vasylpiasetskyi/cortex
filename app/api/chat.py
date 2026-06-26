from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    chat_service = request.app.state.chat_service
    answer = await chat_service.handle(body.session_id, body.message)
    return ChatResponse(answer=answer, session_id=body.session_id)
