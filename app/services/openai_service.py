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

    async def complete(self, messages: list[dict]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except Exception as exc:
            raise OpenAIServiceError(str(exc)) from exc
        tokens = response.usage.total_tokens if response.usage else 0
        logger.debug(f"complete tokens={tokens}")
        return response.choices[0].message.content or ""

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
