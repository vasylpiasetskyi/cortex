import time

from loguru import logger

from app.services.openai_service import OpenAIService


class ChatService:
    def __init__(self, openai_service: OpenAIService) -> None:
        self.openai = openai_service

    async def handle(self, session_id: str, message: str) -> str:
        start = time.monotonic()
        messages = [{"role": "user", "content": message}]
        answer = await self.openai.complete(messages)
        latency_ms = round((time.monotonic() - start) * 1000)
        logger.info(f"session={session_id} latency_ms={latency_ms}")
        return answer
