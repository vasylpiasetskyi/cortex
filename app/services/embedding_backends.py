from abc import ABC, abstractmethod

from openai import AsyncOpenAI

_OPENAI_MODEL = "text-embedding-3-small"


class EmbeddingModelMismatchError(Exception):
    pass


class EmbeddingBackend(ABC):
    vector_size: int

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]: ...


class OpenAIBackend(EmbeddingBackend):
    vector_size = 1536

    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @property
    def model_id(self) -> str:
        return f"openai:{_OPENAI_MODEL}"

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(model=_OPENAI_MODEL, input=text)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(model=_OPENAI_MODEL, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors
