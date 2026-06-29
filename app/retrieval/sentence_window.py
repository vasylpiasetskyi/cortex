from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.models.document import Chunk
from app.retrieval.base import RetrievedChunk, RetrieverStrategy
from app.storage.qdrant import COLLECTION_NAME

WINDOW_SIZE = 2


class SentenceWindowRetriever(RetrieverStrategy):
    def __init__(
        self,
        client: AsyncQdrantClient,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._client = client
        self._session_factory = session_factory

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

        matches = [
            RetrievedChunk(
                chunk_id=point.id,
                document_id=point.payload["document_id"],
                page=point.payload["page"],
                chunk_index=point.payload["chunk_index"],
                score=point.score,
            )
            for point in response.points
        ]

        if not matches:
            return []

        seen_ids: set[int] = set()
        expanded: list[RetrievedChunk] = []

        async with self._session_factory() as session:
            for match in matches:
                result = await session.execute(
                    select(Chunk)
                    .where(
                        Chunk.document_id == match.document_id,
                        Chunk.page == match.page,
                        Chunk.chunk_index >= match.chunk_index - WINDOW_SIZE,
                        Chunk.chunk_index <= match.chunk_index + WINDOW_SIZE,
                    )
                    .order_by(Chunk.chunk_index)
                )
                for chunk in result.scalars():
                    if chunk.id not in seen_ids:
                        seen_ids.add(chunk.id)
                        expanded.append(
                            RetrievedChunk(
                                chunk_id=chunk.id,
                                document_id=chunk.document_id,
                                page=chunk.page,
                                chunk_index=chunk.chunk_index,
                                score=match.score,
                            )
                        )

        return expanded
