from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from app.retrieval.auto_merging import AutoMergingRetriever
from app.retrieval.baseline import BaselineRetriever
from app.retrieval.sentence_window import SentenceWindowRetriever


def make_scored_point(chunk_id: int, document_id: int, session_id: str, page: int, chunk_index: int, score: float):
    point = MagicMock()
    point.id = chunk_id
    point.score = score
    point.payload = {
        "document_id": document_id,
        "session_id": session_id,
        "page": page,
        "chunk_index": chunk_index,
    }
    return point


@pytest.mark.asyncio
async def test_baseline_retrieve_returns_chunks():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [
        make_scored_point(1, 10, "sess-1", 3, 0, 0.91),
        make_scored_point(2, 10, "sess-1", 4, 1, 0.85),
    ]
    mock_client.query_points = AsyncMock(return_value=mock_response)

    retriever = BaselineRetriever(mock_client)
    results = await retriever.retrieve(
        query_vector=[0.1] * 1536,
        session_id="sess-1",
        document_id=10,
        top_k=5,
    )

    assert len(results) == 2
    assert results[0].chunk_id == 1
    assert results[0].page == 3
    assert results[0].score == 0.91
    assert results[1].chunk_id == 2


@pytest.mark.asyncio
async def test_baseline_filters_by_session_id():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points = AsyncMock(return_value=mock_response)

    retriever = BaselineRetriever(mock_client)
    await retriever.retrieve(
        query_vector=[0.1] * 1536,
        session_id="sess-abc",
        document_id=None,
        top_k=5,
    )

    call_kwargs = mock_client.query_points.call_args.kwargs
    # verify session filter is present
    filter_obj = call_kwargs["query_filter"]
    conditions = filter_obj.must
    session_cond = next(c for c in conditions if c.key == "session_id")
    assert session_cond.match.value == "sess-abc"


@pytest.mark.asyncio
async def test_baseline_adds_document_filter_when_provided():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points = AsyncMock(return_value=mock_response)

    retriever = BaselineRetriever(mock_client)
    await retriever.retrieve([0.1] * 1536, "sess-1", document_id=42, top_k=5)

    call_kwargs = mock_client.query_points.call_args.kwargs
    conditions = call_kwargs["query_filter"].must
    doc_cond = next(c for c in conditions if c.key == "document_id")
    assert doc_cond.match.value == 42


@pytest.mark.asyncio
async def test_baseline_no_document_filter_when_none():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points = AsyncMock(return_value=mock_response)

    retriever = BaselineRetriever(mock_client)
    await retriever.retrieve([0.1] * 1536, "sess-1", document_id=None, top_k=5)

    call_kwargs = mock_client.query_points.call_args.kwargs
    conditions = call_kwargs["query_filter"].must
    doc_keys = [c.key for c in conditions if c.key == "document_id"]
    assert doc_keys == []


@pytest.mark.asyncio
async def test_sentence_window_includes_neighboring_chunks(rag_client):
    from datetime import UTC, datetime

    from sqlalchemy import select

    from app.main import app
    from app.models.document import Chunk, Document, DocumentStatus
    from app.retrieval.sentence_window import SentenceWindowRetriever

    async with app.state.session_factory() as session:
        doc = Document(
            session_id="sw-sess",
            filename="f.pdf",
            file_path="x",
            file_size=0,
            status=DocumentStatus.ready,
            created_at=datetime.now(UTC),
        )
        session.add(doc)
        await session.flush()
        doc_id = doc.id
        chunk_objs = [
            Chunk(document_id=doc_id, text=f"chunk {i}", page=0, chunk_index=i, token_count=10)
            for i in range(5)
        ]
        session.add_all(chunk_objs)
        await session.flush()
        chunk_ids = [c.id for c in chunk_objs]
        await session.commit()

    # Qdrant returns the middle chunk (index=2)
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [make_scored_point(chunk_ids[2], doc_id, "sw-sess", 0, 2, 0.9)]
    mock_client.query_points = AsyncMock(return_value=mock_response)

    retriever = SentenceWindowRetriever(mock_client, app.state.session_factory)
    results = await retriever.retrieve([0.1] * 1536, "sw-sess", doc_id, top_k=1)

    result_indices = sorted(r.chunk_index for r in results)
    # WINDOW_SIZE=2: chunks 0,1,2,3,4 all within ±2 of index 2
    assert result_indices == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_sentence_window_deduplicates_overlapping_windows(rag_client):
    from datetime import UTC, datetime

    from app.main import app
    from app.models.document import Chunk, Document, DocumentStatus
    from app.retrieval.sentence_window import SentenceWindowRetriever

    async with app.state.session_factory() as session:
        doc = Document(
            session_id="sw-dedup",
            filename="f.pdf",
            file_path="x",
            file_size=0,
            status=DocumentStatus.ready,
            created_at=datetime.now(UTC),
        )
        session.add(doc)
        await session.flush()
        doc_id = doc.id
        chunk_objs = [
            Chunk(document_id=doc_id, text=f"chunk {i}", page=0, chunk_index=i, token_count=10)
            for i in range(6)
        ]
        session.add_all(chunk_objs)
        await session.flush()
        chunk_ids = [c.id for c in chunk_objs]
        await session.commit()

    # Two matches at index 1 and 3 — their windows overlap at index 2,3
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [
        make_scored_point(chunk_ids[1], doc_id, "sw-dedup", 0, 1, 0.9),
        make_scored_point(chunk_ids[3], doc_id, "sw-dedup", 0, 3, 0.8),
    ]
    mock_client.query_points = AsyncMock(return_value=mock_response)

    retriever = SentenceWindowRetriever(mock_client, app.state.session_factory)
    results = await retriever.retrieve([0.1] * 1536, "sw-dedup", doc_id, top_k=2)

    result_ids = [r.chunk_id for r in results]
    # No duplicates
    assert len(result_ids) == len(set(result_ids))


@pytest.mark.asyncio
async def test_sentence_window_neighbors_inherit_match_score(rag_client):
    from datetime import UTC, datetime

    from app.main import app
    from app.models.document import Chunk, Document, DocumentStatus
    from app.retrieval.sentence_window import SentenceWindowRetriever

    async with app.state.session_factory() as session:
        doc = Document(
            session_id="sw-score",
            filename="f.pdf",
            file_path="x",
            file_size=0,
            status=DocumentStatus.ready,
            created_at=datetime.now(UTC),
        )
        session.add(doc)
        await session.flush()
        doc_id = doc.id
        chunk_objs = [
            Chunk(document_id=doc_id, text=f"chunk {i}", page=0, chunk_index=i, token_count=10)
            for i in range(3)
        ]
        session.add_all(chunk_objs)
        await session.flush()
        chunk_ids = [c.id for c in chunk_objs]
        await session.commit()

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [make_scored_point(chunk_ids[1], doc_id, "sw-score", 0, 1, 0.77)]
    mock_client.query_points = AsyncMock(return_value=mock_response)

    retriever = SentenceWindowRetriever(mock_client, app.state.session_factory)
    results = await retriever.retrieve([0.1] * 1536, "sw-score", doc_id, top_k=1)

    # All returned chunks (matched + neighbors) carry the parent match score
    assert all(r.score == 0.77 for r in results)


@pytest.mark.asyncio
async def test_auto_merging_raises_not_implemented():
    retriever = AutoMergingRetriever(MagicMock())
    with pytest.raises(NotImplementedError):
        await retriever.retrieve([0.1] * 1536, "sess-1", None, 5)
