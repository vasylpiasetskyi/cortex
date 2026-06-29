import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_valid_pdf_returns_202(rag_client: AsyncClient):
    fake_pdf = b"%PDF-1.4 fake content"
    response = await rag_client.post(
        "/documents",
        data={"session_id": "sess-upload"},
        files={"file": ("contract.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "contract.pdf"
    assert body["status"] == "indexing"
    assert "id" in body


@pytest.mark.asyncio
async def test_upload_non_pdf_returns_422(rag_client: AsyncClient):
    response = await rag_client.post(
        "/documents",
        data={"session_id": "sess-upload"},
        files={"file": ("notes.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_upload_too_large_returns_422(rag_client: AsyncClient):
    large_file = b"%PDF-1.4 " + b"x" * (21 * 1024 * 1024)  # 21 MB
    response = await rag_client.post(
        "/documents",
        data={"session_id": "sess-upload"},
        files={"file": ("big.pdf", large_file, "application/pdf")},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_documents_returns_only_own_session(rag_client: AsyncClient):
    fake_pdf = b"%PDF-1.4 fake"

    # Upload for session A
    await rag_client.post(
        "/documents",
        data={"session_id": "sess-A"},
        files={"file": ("doc_a.pdf", fake_pdf, "application/pdf")},
    )
    # Upload for session B
    await rag_client.post(
        "/documents",
        data={"session_id": "sess-B"},
        files={"file": ("doc_b.pdf", fake_pdf, "application/pdf")},
    )

    resp_a = await rag_client.get("/documents", params={"session_id": "sess-A"})
    assert resp_a.status_code == 200
    filenames_a = [d["filename"] for d in resp_a.json()]
    assert "doc_a.pdf" in filenames_a
    assert "doc_b.pdf" not in filenames_a


@pytest.mark.asyncio
async def test_delete_document_returns_204_then_404(rag_client: AsyncClient):
    fake_pdf = b"%PDF-1.4 fake"
    upload_resp = await rag_client.post(
        "/documents",
        data={"session_id": "sess-del"},
        files={"file": ("todelete.pdf", fake_pdf, "application/pdf")},
    )
    doc_id = upload_resp.json()["id"]

    delete_resp = await rag_client.delete(f"/documents/{doc_id}", params={"session_id": "sess-del"})
    assert delete_resp.status_code == 204

    get_resp = await rag_client.get(f"/documents/{doc_id}", params={"session_id": "sess-del"})
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_reindex_returns_202(rag_client: AsyncClient):
    from app.main import app
    from datetime import UTC, datetime
    from app.models.document import Document, DocumentStatus

    async with app.state.session_factory() as session:
        doc = Document(
            session_id="reindex-sess",
            filename="doc.pdf",
            file_path="uploads/reindex-test.pdf",
            file_size=100,
            status=DocumentStatus.ready,
            embedding_model="local:some-model",
            created_at=datetime.now(UTC),
        )
        session.add(doc)
        await session.flush()
        doc_id = doc.id
        await session.commit()

    resp = await rag_client.post(
        f"/documents/{doc_id}/reindex",
        params={"session_id": "reindex-sess"},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "indexing"
    assert body["id"] == doc_id


@pytest.mark.asyncio
async def test_index_document_creates_chunks_and_sets_ready(rag_client: AsyncClient):
    from datetime import UTC, datetime

    from sqlalchemy import select

    from app.main import app
    from app.models.document import Chunk, Document, DocumentStatus

    async with app.state.session_factory() as session:
        doc = Document(
            session_id="idx-sess",
            filename="sample.pdf",
            file_path="uploads/sample.pdf",
            file_size=1000,
            status=DocumentStatus.indexing,
            created_at=datetime.now(UTC),
        )
        session.add(doc)
        await session.flush()
        doc_id = doc.id
        await session.commit()

    sample_pages = [
        {"page": 0, "text": "Hello world " * 100},
        {"page": 1, "text": "Another page " * 100},
    ]

    doc_svc = app.state.doc_service
    original_parse = doc_svc._parser.parse
    doc_svc._parser.parse = lambda path: sample_pages
    try:
        await doc_svc._index_document(doc_id)
    finally:
        doc_svc._parser.parse = original_parse

    async with app.state.session_factory() as session:
        doc = await session.get(Document, doc_id)
        assert doc.status == DocumentStatus.ready
        assert doc.embedding_model is not None

        result = await session.execute(select(Chunk).where(Chunk.document_id == doc_id))
        chunks = result.scalars().all()
        assert len(chunks) > 0

    assert doc_svc._qdrant.upsert.called


@pytest.mark.asyncio
async def test_reindex_wrong_session_returns_404(rag_client: AsyncClient):
    from app.main import app
    from datetime import UTC, datetime
    from app.models.document import Document, DocumentStatus

    async with app.state.session_factory() as session:
        doc = Document(
            session_id="owner-reindex",
            filename="doc.pdf",
            file_path="uploads/owner.pdf",
            file_size=100,
            status=DocumentStatus.ready,
            embedding_model="openai:text-embedding-3-small",
            created_at=datetime.now(UTC),
        )
        session.add(doc)
        await session.flush()
        doc_id = doc.id
        await session.commit()

    resp = await rag_client.post(
        f"/documents/{doc_id}/reindex",
        params={"session_id": "other-sess"},
    )
    assert resp.status_code == 404
