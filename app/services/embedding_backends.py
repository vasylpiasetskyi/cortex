import asyncio
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


class LocalBackend(EmbeddingBackend):
    """Sentence-transformers backend for local inference."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for LocalBackend. "
                "Install it with: uv add sentence-transformers"
            ) from e

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        sample = self._model.encode("probe")
        self.vector_size: int = len(sample)

    @property
    def model_id(self) -> str:
        return f"local:{self._model_name}"

    async def embed(self, text: str) -> list[float]:
        vector = await asyncio.to_thread(self._model.encode, text)
        return vector.tolist()

    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        vectors = await asyncio.to_thread(self._model.encode, texts)
        return [v.tolist() for v in vectors]


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
