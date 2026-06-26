import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


def make_mock_response(content: str) -> MagicMock:
    mock_usage = MagicMock()
    mock_usage.total_tokens = 10
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    return mock_response


def make_mock_openai():
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            make_mock_response("mocked response"),
            make_mock_response("Your name is Alice."),
            make_mock_response("mocked response"),
            make_mock_response("mocked response"),
            make_mock_response("mocked response"),
        ]
    )
    return mock_client


@pytest_asyncio.fixture
async def client():
    with patch("app.services.openai_service.AsyncOpenAI", return_value=make_mock_openai()):
        from app.main import app
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c
