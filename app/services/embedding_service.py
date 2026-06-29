from app.services.embedding_backends import EmbeddingBackend


class EmbeddingService:
    def __init__(self, backend: EmbeddingBackend) -> None:
        self._backend = backend

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @property
    def vector_size(self) -> int:
        return self._backend.vector_size

    async def embed(self, text: str) -> list[float]:
        return await self._backend.embed(text)

    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        return await self._backend.embed_batch(texts, batch_size)
