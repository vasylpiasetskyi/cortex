from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.retrieval.base import RetrievedChunk, RetrieverStrategy
from app.storage.qdrant import COLLECTION_NAME


class BaselineRetriever(RetrieverStrategy):
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def retrieve(
        self,
        query_vector: list[float],
        session_id: str,
        document_id: int | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        conditions = [FieldCondition(key="session_id", match=MatchValue(value=session_id))]
        if document_id is not None:
            conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            )

        response = await self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(must=conditions),
            limit=top_k,
        )

        return [
            RetrievedChunk(
                chunk_id=point.id,
                document_id=point.payload["document_id"],
                page=point.payload["page"],
                chunk_index=point.payload["chunk_index"],
                score=point.score,
            )
            for point in response.points
        ]
