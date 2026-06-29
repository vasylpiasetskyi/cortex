from unittest.mock import AsyncMock, MagicMock

import pytest

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
async def test_sentence_window_raises_not_implemented():
    retriever = SentenceWindowRetriever(MagicMock())
    with pytest.raises(NotImplementedError):
        await retriever.retrieve([0.1] * 1536, "sess-1", None, 5)


@pytest.mark.asyncio
async def test_auto_merging_raises_not_implemented():
    retriever = AutoMergingRetriever(MagicMock())
    with pytest.raises(NotImplementedError):
        await retriever.retrieve([0.1] * 1536, "sess-1", None, 5)
