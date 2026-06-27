import time
from typing import AsyncGenerator

from loguru import logger

from app.services.conversation_service import ConversationService
from app.services.openai_service import CompletionResult, OpenAIService
from app.tools.registry import TOOLS, execute_tool


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

        result = await self.openai.complete_with_tools(history, TOOLS)

        while result.finish_reason == "tool_calls" and result.tool_calls:
            history.append({
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in result.tool_calls
                ],
            })
            for tc in result.tool_calls:
                tool_result = execute_tool(tc.function.name, tc.function.arguments)
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
            result = await self.openai.complete_with_tools(history, TOOLS)

        answer = result.content or ""

        await self.conv.save_message(session_id, "user", message)
        await self.conv.save_message(session_id, "assistant", answer)

        latency_ms = round((time.monotonic() - start) * 1000)
        logger.info(f"session={session_id} tokens={result.tokens} latency_ms={latency_ms}")
        return answer

    async def handle_stream(self, session_id: str, message: str) -> AsyncGenerator[str, None]:
        start = time.monotonic()

        history = await self.conv.get_history(session_id)
        history.append({"role": "user", "content": message})

        full_response = ""

        while True:
            result: CompletionResult | None = None
            async for item in self.openai.stream_with_tools_support(history, TOOLS):
                if isinstance(item, CompletionResult):
                    result = item
                else:
                    full_response += item
                    yield item

            if result is None or result.finish_reason != "tool_calls" or not result.tool_calls:
                break

            history.append({
                "role": "assistant",
                "content": result.content,
                "tool_calls": result.tool_calls,
            })
            for tc in result.tool_calls:
                tool_result = execute_tool(tc["function"]["name"], tc["function"]["arguments"])
                history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

        await self.conv.save_message(session_id, "user", message)
        await self.conv.save_message(session_id, "assistant", full_response)

        latency_ms = round((time.monotonic() - start) * 1000)
        logger.info(f"session={session_id} latency_ms={latency_ms}")
