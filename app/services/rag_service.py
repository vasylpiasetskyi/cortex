from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.document import Chunk, DocumentStatus
from app.retrieval.auto_merging import AutoMergingRetriever
from app.retrieval.base import RetrievedChunk, RetrieverStrategy
from app.retrieval.baseline import BaselineRetriever
from app.retrieval.sentence_window import SentenceWindowRetriever
from app.services.document_service import DocumentService
from app.services.embedding_backends import EmbeddingModelMismatchError
from app.services.embedding_service import EmbeddingService
from app.services.openai_service import OpenAIService

TOP_K = 5


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    context_parts = [f"[Page {c.page + 1}]\n{c.text}" for c in chunks]
    context = "\n\n".join(context_parts)
    system = (
        "Answer ONLY using the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


class RAGService:
    def __init__(
        self,
        openai_svc: OpenAIService,
        embedding_svc: EmbeddingService,
        doc_svc: DocumentService,
        session_factory: async_sessionmaker[AsyncSession],
        qdrant_client,
    ) -> None:
        self._openai = openai_svc
        self._embedder = embedding_svc
        self._doc_svc = doc_svc
        self._session_factory = session_factory
        self._strategies: dict[str, RetrieverStrategy] = {
            "baseline": BaselineRetriever(qdrant_client),
            "sentence_window": SentenceWindowRetriever(qdrant_client, session_factory),
            "auto_merging": AutoMergingRetriever(qdrant_client),
        }

    async def ask(
        self,
        session_id: str,
        question: str,
        document_id: int | None,
        strategy: str,
    ) -> dict:
        if not question.strip():
            raise ValueError("question must not be empty")

        if strategy not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        if document_id is not None:
            doc = await self._doc_svc.get_document(document_id, session_id)
            if doc is None:
                raise LookupError(f"Document {document_id} not found")
            if doc.status != DocumentStatus.ready:
                raise PermissionError(doc.status)
            if doc.embedding_model != self._embedder.model_id:
                raise EmbeddingModelMismatchError(
                    f"Document indexed with '{doc.embedding_model}', "
                    f"current backend is '{self._embedder.model_id}'"
                )

        query_vector = await self._embedder.embed(question)

        retriever = self._strategies[strategy]
        retrieved = await retriever.retrieve(query_vector, session_id, document_id, TOP_K)

        if retrieved:
            chunk_ids = [rc.chunk_id for rc in retrieved]
            async with self._session_factory() as session:
                result = await session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
                chunks_by_id = {c.id: c.text for c in result.scalars()}
            for rc in retrieved:
                rc.text = chunks_by_id.get(rc.chunk_id, "")

        messages = _build_prompt(question, retrieved)
        completion = await self._openai.complete_with_tools(messages, [])
        answer = completion.content or ""

        logger.bind(session_id=session_id, strategy=strategy, sources=len(retrieved)).info(
            "rag complete"
        )

        return {
            "answer": answer,
            "sources": [
                {"page": rc.page, "chunk_index": rc.chunk_index, "score": round(rc.score, 4)}
                for rc in retrieved
            ],
            "strategy": strategy,
        }
