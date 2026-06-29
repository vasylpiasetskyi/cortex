from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from app.models.document import Chunk, Document, DocumentStatus


async def seed_ready_document(app, session_id: str) -> tuple[int, int]:
    """Insert Document + Chunk, return (doc_id, chunk_id). Uses autoincrement — safe to call multiple times."""
    async with app.state.session_factory() as session:
        doc = Document(
            session_id=session_id,
            filename="test.pdf",
            file_path="uploads/test.pdf",
            file_size=1024,
            status=DocumentStatus.ready,
            embedding_model="openai:text-embedding-3-small",
            created_at=datetime.now(UTC),
        )
        session.add(doc)
        await session.flush()  # populate doc.id
        chunk = Chunk(
            document_id=doc.id,
            text="The cancellation policy allows 30 days notice.",
            page=4,
            chunk_index=0,
            token_count=10,
        )
        session.add(chunk)
        await session.flush()  # populate chunk.id
        doc_id, chunk_id = doc.id, chunk.id
        await session.commit()
    return doc_id, chunk_id


def _make_point(chunk_id, document_id, session_id, page, chunk_index, score):
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
async def test_ask_returns_answer_and_sources(rag_client: AsyncClient):
    from app.main import app

    doc_id, chunk_id = await seed_ready_document(app, session_id="ask-sess")
    mock_response = MagicMock()
    mock_response.points = [
        _make_point(chunk_id=chunk_id, document_id=doc_id, session_id="ask-sess", page=4, chunk_index=0, score=0.91)
    ]
    app.state.qdrant.query_points = AsyncMock(return_value=mock_response)

    resp = await rag_client.post("/ask", json={
        "session_id": "ask-sess",
        "question": "What is the cancellation policy?",
        "document_id": doc_id,
        "strategy": "baseline",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert len(body["sources"]) == 1
    assert body["sources"][0]["page"] == 4
    assert body["strategy"] == "baseline"


@pytest.mark.asyncio
async def test_ask_without_document_id_searches_all_session_docs(rag_client: AsyncClient):
    from app.main import app

    await seed_ready_document(app, session_id="ask-all")
    mock_response = MagicMock()
    mock_response.points = []
    app.state.qdrant.query_points = AsyncMock(return_value=mock_response)

    resp = await rag_client.post("/ask", json={
        "session_id": "ask-all",
        "question": "anything?",
    })
    assert resp.status_code == 200
    call_kwargs = app.state.qdrant.query_points.call_args.kwargs
    conditions = call_kwargs["query_filter"].must
    assert not any(c.key == "document_id" for c in conditions)


@pytest.mark.asyncio
async def test_ask_wrong_session_returns_404(rag_client: AsyncClient):
    from app.main import app

    doc_id, _ = await seed_ready_document(app, session_id="owner-sess")
    resp = await rag_client.post("/ask", json={
        "session_id": "other-sess",
        "question": "test",
        "document_id": doc_id,
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ask_indexing_document_returns_409(rag_client: AsyncClient):
    from app.main import app

    async with app.state.session_factory() as session:
        doc = Document(
            session_id="idx-sess",
            filename="f.pdf",
            file_path="uploads/f.pdf",
            file_size=100,
            status=DocumentStatus.indexing,
            created_at=datetime.now(UTC),
        )
        session.add(doc)
        await session.flush()
        doc_id = doc.id
        await session.commit()

    resp = await rag_client.post("/ask", json={
        "session_id": "idx-sess",
        "question": "test",
        "document_id": doc_id,
    })
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_ask_sentence_window_strategy_returns_501(rag_client: AsyncClient):
    from app.main import app

    doc_id, _ = await seed_ready_document(app, session_id="strat-sess")
    resp = await rag_client.post("/ask", json={
        "session_id": "strat-sess",
        "question": "test",
        "document_id": doc_id,
        "strategy": "sentence_window",
    })
    assert resp.status_code == 501


@pytest.mark.asyncio
async def test_ask_empty_question_returns_422(rag_client: AsyncClient):
    resp = await rag_client.post("/ask", json={
        "session_id": "sess",
        "question": "",
    })
    assert resp.status_code == 422
