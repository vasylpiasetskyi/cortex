import time
from typing import AsyncGenerator

from loguru import logger

from app.services.conversation_service import ConversationService
from app.services.openai_service import OpenAIService


class ChatService:
    def __init__(
        self,
        openai_service: OpenAIService,
        conversation_service: ConversationService,
    ) -> None:
        self.openai = openai_service
        self.conv = conversation_service

    async def handle(self, session_id: str, message: str) -> str:
        start = time.monotonic()

        history = await self.conv.get_history(session_id)
        history.append({"role": "user", "content": message})

        answer = await self.openai.complete(history)

        await self.conv.save_message(session_id, "user", message)
        await self.conv.save_message(session_id, "assistant", answer)

        latency_ms = round((time.monotonic() - start) * 1000)
        logger.info(f"session={session_id} latency_ms={latency_ms}")
        return answer

    async def handle_stream(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        history = await self.conv.get_history(session_id)
        history.append({"role": "user", "content": message})
        await self.conv.save_message(session_id, "user", message)

        full_response = ""
        async for chunk in self.openai.stream(history):
            full_response += chunk
            yield chunk

        await self.conv.save_message(session_id, "assistant", full_response)
