from dataclasses import dataclass
from typing import AsyncGenerator, TypeVar, Type

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel as PydanticModel

T = TypeVar("T", bound=PydanticModel)


class OpenAIServiceError(Exception):
    pass


@dataclass
class CompletionResult:
    content: str | None
    finish_reason: str
    tool_calls: list | None
    tokens: int = 0


class OpenAIService:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def stream(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )
        except Exception as exc:
            raise OpenAIServiceError(str(exc)) from exc
        try:
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise OpenAIServiceError(str(exc)) from exc

    async def stream_with_tools_support(
        self, messages: list[dict], tools: list[dict]
    ) -> AsyncGenerator[str | CompletionResult, None]:
        """Yields str tokens during streaming, then a final CompletionResult with tool_calls if needed."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=True,
            )
        except Exception as exc:
            raise OpenAIServiceError(str(exc)) from exc

        tool_calls_acc: dict[int, dict] = {}
        content_acc = ""
        finish_reason = "stop"

        try:
            async for chunk in response:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta.content:
                    content_acc += delta.content
                    yield delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls_acc[idx]["function"]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc.function.arguments
        except Exception as exc:
            raise OpenAIServiceError(str(exc)) from exc

        yield CompletionResult(
            content=content_acc,
            finish_reason=finish_reason,
            tool_calls=list(tool_calls_acc.values()) if tool_calls_acc else None,
        )

    async def complete_with_tools(
        self, messages: list[dict], tools: list[dict]
    ) -> CompletionResult:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            )
        except Exception as exc:
            raise OpenAIServiceError(str(exc)) from exc
        choice = response.choices[0]
        return CompletionResult(
            content=choice.message.content,
            finish_reason=choice.finish_reason,
            tool_calls=choice.message.tool_calls,
            tokens=response.usage.total_tokens if response.usage else 0,
        )

    async def extract_structured(self, prompt: str, schema: Type[T]) -> T:
        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
            )
        except Exception as exc:
            raise OpenAIServiceError(str(exc)) from exc
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise OpenAIServiceError("Model returned no structured output")
        return parsed
