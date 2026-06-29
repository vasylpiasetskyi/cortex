from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.models.document import Chunk
from app.retrieval.base import RetrievedChunk, RetrieverStrategy
from app.storage.qdrant import COLLECTION_NAME

MERGE_RATIO = 0.5


class AutoMergingRetriever(RetrieverStrategy):
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

        # Group matches by (document_id, page)
        groups: dict[tuple[int, int], list[RetrievedChunk]] = defaultdict(list)
        for m in matches:
            groups[(m.document_id, m.page)].append(m)

        result: list[RetrievedChunk] = []

        async with self._session_factory() as session:
            for (doc_id, page), group_matches in groups.items():
                total_on_page = await session.scalar(
                    select(func.count(Chunk.id)).where(
                        Chunk.document_id == doc_id,
                        Chunk.page == page,
                    )
                )
                ratio = len(group_matches) / total_on_page if total_on_page else 0

                if ratio >= MERGE_RATIO:
                    # Merge: replace matched chunks with the full page text
                    page_chunks = await session.execute(
                        select(Chunk)
                        .where(Chunk.document_id == doc_id, Chunk.page == page)
                        .order_by(Chunk.chunk_index)
                    )
                    merged_text = " ".join(c.text for c in page_chunks.scalars())
                    best = max(group_matches, key=lambda m: m.score)
                    result.append(
                        RetrievedChunk(
                            chunk_id=best.chunk_id,
                            document_id=doc_id,
                            page=page,
                            chunk_index=best.chunk_index,
                            score=best.score,
                            text=merged_text,
                        )
                    )
                else:
                    result.extend(group_matches)

        return result
