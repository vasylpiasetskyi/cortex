from unittest.mock import AsyncMock, MagicMock
import pytest
from app.services.embedding_backends import OpenAIBackend
from app.services.embedding_service import EmbeddingService


def make_embed_client(vector: list[float] | None = None):
    vector = vector or [0.1] * 1536
    mock_client = MagicMock()

    async def create_side_effect(*args, **kwargs):
        inp = kwargs.get("input", [])
        if isinstance(inp, str):
            inp = [inp]
        response = MagicMock()
        response.data = [MagicMock(embedding=vector) for _ in inp]
        return response

    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=create_side_effect)
    return mock_client


@pytest.mark.asyncio
async def test_embed_returns_vector():
    svc = EmbeddingService(OpenAIBackend(make_embed_client()))
    result = await svc.embed("hello world")
    assert len(result) == 1536
    assert result[0] == 0.1


@pytest.mark.asyncio
async def test_embed_batch_returns_one_vector_per_text():
    svc = EmbeddingService(OpenAIBackend(make_embed_client()))
    texts = ["text one", "text two", "text three"]
    result = await svc.embed_batch(texts)
    assert len(result) == 3
    assert all(len(v) == 1536 for v in result)


@pytest.mark.asyncio
async def test_embed_batch_respects_batch_size():
    mock_client = make_embed_client()
    svc = EmbeddingService(OpenAIBackend(mock_client))
    texts = [f"text {i}" for i in range(250)]
    await svc.embed_batch(texts, batch_size=100)
    assert mock_client.embeddings.create.call_count == 3


@pytest.mark.asyncio
async def test_embed_batch_preserves_order():
    vectors = [[float(i)] * 1536 for i in range(5)]
    mock_client = MagicMock()
    call_count = [0]

    async def create_side_effect(*args, **kwargs):
        inp = kwargs.get("input", [])
        response = MagicMock()
        response.data = [MagicMock(embedding=vectors[call_count[0] + j]) for j in range(len(inp))]
        call_count[0] += len(inp)
        return response

    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=create_side_effect)

    svc = EmbeddingService(OpenAIBackend(mock_client))
    result = await svc.embed_batch([f"t{i}" for i in range(5)], batch_size=3)
    assert len(result) == 5
    assert result[0][0] == 0.0
    assert result[4][0] == 4.0
