import itertools
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


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

    if responses is None:
        # Cycle "mocked response" indefinitely
        response_iter = itertools.cycle(["mocked response"])
    else:
        response_iter = iter(responses)

    async def create_side_effect(*args, **kwargs):
        if kwargs.get("stream"):
            async def mock_stream():
                for token in ["hello", " ", "world"]:
                    chunk = MagicMock()
                    chunk.choices[0].delta.content = token
                    yield chunk
            return mock_stream()
        else:
            # Check if user messages in history establish a fact we can recall
            messages = kwargs.get("messages") or (args[0] if args else [])
            import re
            # Only scan user-role messages (not assistant responses) for "named X" facts
            user_history = [
                m.get("content", "")
                for m in messages[:-1]  # exclude current question
                if m.get("role") == "user"
            ]
            named_match = None
            for content in user_history:
                m = re.search(r"\bnamed\s+([A-Z][a-z]+)\b", content)
                if m:
                    named_match = m.group(1)
            if named_match:
                return make_response(f"Your dog is named {named_match}.")
            return make_response(next(response_iter))

    mock_client.chat.completions.create = AsyncMock(side_effect=create_side_effect)

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


@pytest_asyncio.fixture
async def client():
    with patch("app.services.openai_service.AsyncOpenAI", return_value=make_mock_openai()):
        from app.main import app
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c
