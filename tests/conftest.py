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
        return make_response(next(response_iter))

    mock_client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    return mock_client


@pytest_asyncio.fixture
async def client():
    with patch("app.services.openai_service.AsyncOpenAI", return_value=make_mock_openai()):
        from app.main import app
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c
