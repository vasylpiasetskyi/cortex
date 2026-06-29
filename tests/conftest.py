import itertools
from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient


def make_mock_openai(responses=None):
    mock_client = MagicMock()
    mock_usage = MagicMock()
    mock_usage.total_tokens = 10

    def make_response(content):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        return mock_response

    response_iter = itertools.cycle(["mocked response"]) if responses is None else iter(responses)

    async def create_side_effect(*args, **kwargs):
        if kwargs.get("stream"):

            async def mock_stream():
                for token in ["hello", " ", "world"]:
                    chunk = MagicMock()
                    chunk.choices[0].delta.content = token
                    yield chunk

            return mock_stream()
        else:
            messages = kwargs.get("messages") or (args[0] if args else [])
            last_user_msg = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
            )
            if "weather" in last_user_msg.lower() and not any(
                m.get("role") == "tool" for m in messages
            ):
                mock_tool_call = MagicMock()
                mock_tool_call.id = "call_123"
                mock_tool_call.function.name = "get_weather"
                mock_tool_call.function.arguments = '{"city": "Warsaw"}'
                mock_choice = MagicMock()
                mock_choice.message.content = None
                mock_choice.finish_reason = "tool_calls"
                mock_choice.message.tool_calls = [mock_tool_call]
                mock_response = MagicMock()
                mock_response.choices = [mock_choice]
                mock_response.usage.total_tokens = 15
                return mock_response
            else:
                import re

                user_history = [
                    m.get("content", "")
                    for m in messages[:-1]
                    if m.get("role") == "user"
                ]
                named_match = None
                for content in user_history:
                    m = re.search(r"\bnamed\s+([A-Z][a-z]+)\b", content)
                    if m:
                        named_match = m.group(1)
                if named_match:
                    return make_response(f"Your dog is named {named_match}.")
                tool_results = [m for m in messages if m.get("role") == "tool"]
                if tool_results:
                    tool_content = tool_results[-1].get("content", "")
                    return make_response(f"The weather information: {tool_content}")
                return make_response(next(response_iter))

    mock_client.chat.completions.create = AsyncMock(side_effect=create_side_effect)

    # embeddings mock — returns one vector per input text
    async def embed_side_effect(*args, **kwargs):
        inp = kwargs.get("input", [])
        if isinstance(inp, str):
            inp = [inp]
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1] * 1536) for _ in inp]
        return response

    mock_client.embeddings = MagicMock()
    mock_client.embeddings.create = AsyncMock(side_effect=embed_side_effect)

    mock_person = MagicMock()
    mock_person.name = "John Smith"
    mock_person.age = 35
    mock_parse_choice = MagicMock()
    mock_parse_choice.message.parsed = mock_person
    mock_parse_response = MagicMock()
    mock_parse_response.choices = [mock_parse_choice]
    mock_client.beta = MagicMock()
    mock_client.beta.chat = MagicMock()
    mock_client.beta.chat.completions = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_parse_response)

    return mock_client


def make_mock_qdrant():
    mock_qdrant = MagicMock()

    # get_collections returns object with empty collections list
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_qdrant.get_collections = AsyncMock(return_value=mock_collections)
    mock_qdrant.create_collection = AsyncMock(return_value=None)

    mock_collection_info = MagicMock()
    mock_collection_info.config.params.vectors.size = 1536
    mock_qdrant.get_collection = AsyncMock(return_value=mock_collection_info)

    # query_points returns empty points list by default
    mock_points_response = MagicMock()
    mock_points_response.points = []
    mock_qdrant.query_points = AsyncMock(return_value=mock_points_response)

    # upsert / delete return None
    mock_qdrant.upsert = AsyncMock(return_value=None)
    mock_qdrant.delete = AsyncMock(return_value=None)

    # close
    mock_qdrant.close = AsyncMock(return_value=None)

    return mock_qdrant


@pytest_asyncio.fixture
async def client():
    with (
        patch("app.services.openai_service.AsyncOpenAI", return_value=make_mock_openai()),
        patch("app.storage.qdrant.AsyncQdrantClient", return_value=make_mock_qdrant()),
    ):
        from app.main import app

        async with app.router.lifespan_context(app), AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


@pytest_asyncio.fixture
async def rag_client():
    """Client with both OpenAI (including embeddings) and Qdrant mocked."""
    with (
        patch("app.services.openai_service.AsyncOpenAI", return_value=make_mock_openai()),
        patch("app.storage.qdrant.AsyncQdrantClient", return_value=make_mock_qdrant()),
    ):
        from app.main import app

        async with app.router.lifespan_context(app), AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c
