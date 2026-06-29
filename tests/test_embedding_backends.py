from unittest.mock import AsyncMock, MagicMock
import pytest
from app.services.embedding_backends import OpenAIBackend

def make_openai_client(vector=None):
    vector = vector or [0.1] * 1536
    mock_client = MagicMock()
    async def embed_side(*args, **kwargs):
        inp = kwargs.get("input", [])
        if isinstance(inp, str):
            inp = [inp]
        response = MagicMock()
        response.data = [MagicMock(embedding=vector) for _ in inp]
        return response
    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=embed_side)
    return mock_client

@pytest.mark.asyncio
async def test_openai_backend_embed_returns_vector():
    backend = OpenAIBackend(make_openai_client())
    result = await backend.embed("hello")
    assert len(result) == 1536
    assert result[0] == 0.1

@pytest.mark.asyncio
async def test_openai_backend_embed_batch():
    backend = OpenAIBackend(make_openai_client())
    result = await backend.embed_batch(["a", "b", "c"])
    assert len(result) == 3
    assert all(len(v) == 1536 for v in result)

@pytest.mark.asyncio
async def test_openai_backend_respects_batch_size():
    client = make_openai_client()
    backend = OpenAIBackend(client)
    texts = [f"t{i}" for i in range(250)]
    await backend.embed_batch(texts, batch_size=100)
    assert client.embeddings.create.call_count == 3

def test_openai_backend_model_id():
    backend = OpenAIBackend(make_openai_client())
    assert backend.model_id == "openai:text-embedding-3-small"

def test_openai_backend_vector_size():
    backend = OpenAIBackend(make_openai_client())
    assert backend.vector_size == 1536
