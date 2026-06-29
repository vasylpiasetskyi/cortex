from qdrant_client import AsyncQdrantClient

from app.retrieval.base import RetrievedChunk, RetrieverStrategy


class SentenceWindowRetriever(RetrieverStrategy):
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def retrieve(
        self,
        query_vector: list[float],
        session_id: str,
        document_id: int | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError("SentenceWindowRetriever is not yet implemented")
