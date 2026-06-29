import asyncio
import contextlib
import uuid
from pathlib import Path

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue, PointStruct
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.document import Chunk, Document, DocumentStatus
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.parser_service import ParserService
from app.storage.qdrant import COLLECTION_NAME

UPLOAD_DIR = Path("uploads")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


class DocumentService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        parser: ParserService,
        chunker: ChunkingService,
        embedder: EmbeddingService,
        qdrant: AsyncQdrantClient,
    ) -> None:
        self._session_factory = session_factory
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._qdrant = qdrant
        UPLOAD_DIR.mkdir(exist_ok=True)

    async def create_document(
        self, session_id: str, filename: str, file_bytes: bytes
    ) -> Document:
        file_path = UPLOAD_DIR / f"{uuid.uuid4()}.pdf"
        file_path.write_bytes(file_bytes)

        async with self._session_factory() as session:
            doc = Document(
                session_id=session_id,
                filename=filename,
                file_path=str(file_path),
                file_size=len(file_bytes),
                status=DocumentStatus.indexing,
            )
            session.add(doc)
            await session.commit()
            await session.refresh(doc)
            return doc

    async def _index_document(self, document_id: int) -> None:
        try:
            async with self._session_factory() as session:
                doc = await session.get(Document, document_id)
                if doc is None:
                    return

                pages = await asyncio.to_thread(self._parser.parse, doc.file_path)
                chunk_dicts = await asyncio.to_thread(self._chunker.chunk, pages)

                chunks = [
                    Chunk(
                        document_id=document_id,
                        text=c["text"],
                        page=c["page"],
                        chunk_index=c["chunk_index"],
                        token_count=c["token_count"],
                    )
                    for c in chunk_dicts
                ]
                session.add_all(chunks)
                await session.flush()  # populate chunk IDs

                texts = [c.text for c in chunks]
                vectors = await self._embedder.embed_batch(texts)

                points = [
                    PointStruct(
                        id=chunk.id,
                        vector=vector,
                        payload={
                            "document_id": document_id,
                            "session_id": doc.session_id,
                            "page": chunk.page,
                            "chunk_index": chunk.chunk_index,
                            "chunk_id": chunk.id,
                        },
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
                await self._qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

                doc.embedding_model = self._embedder.model_id
                doc.status = DocumentStatus.ready
                await session.commit()
                logger.bind(document_id=document_id).info("document indexed")

        except Exception as exc:
            logger.bind(document_id=document_id).error(f"indexing failed: {exc}")
            async with self._session_factory() as session:
                doc = await session.get(Document, document_id)
                if doc:
                    doc.status = DocumentStatus.error
                    doc.error_msg = str(exc)
                    await session.commit()
            await self._cleanup_partial(document_id)

    async def _cleanup_partial(self, document_id: int) -> None:
        with contextlib.suppress(Exception):
            await self._qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                    )
                ),
            )

    async def list_documents(self, session_id: str) -> list[Document]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Document).where(Document.session_id == session_id)
            )
            return list(result.scalars().all())

    async def get_document(self, document_id: int, session_id: str) -> Document | None:
        async with self._session_factory() as session:
            doc = await session.get(Document, document_id)
            if doc is None or doc.session_id != session_id:
                return None
            return doc

    async def delete_document(self, document_id: int, session_id: str) -> bool:
        async with self._session_factory() as session:
            doc = await session.get(Document, document_id)
            if doc is None or doc.session_id != session_id:
                return False
            await session.delete(doc)
            await session.commit()

        await self._cleanup_partial(document_id)
        return True

    async def start_reindex(self, document_id: int, session_id: str) -> Document | None:
        async with self._session_factory() as session:
            doc = await session.get(Document, document_id)
            if doc is None or doc.session_id != session_id:
                return None
            await self._cleanup_partial(document_id)
            doc.status = DocumentStatus.indexing
            doc.embedding_model = None
            await session.commit()
            await session.refresh(doc)
            return doc

    async def get_chunks_by_ids(self, chunk_ids: list[int]) -> list:
        if not chunk_ids:
            return []
        async with self._session_factory() as session:
            result = await session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
            return list(result.scalars().all())
