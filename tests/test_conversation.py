import pytest_asyncio
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.config import settings
from app.models.db import Base
from app.services.conversation_service import ConversationService


@pytest_asyncio.fixture
async def redis_client():
    client = Redis.from_url(settings.redis_url, decode_responses=True)
    yield client
    await client.flushdb()
    await client.aclose()


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def conv_service(redis_client, session_factory):
    return ConversationService(redis_client, session_factory)


async def test_empty_history(conv_service):
    history = await conv_service.get_history("brand-new-session")
    assert history == []


async def test_save_and_retrieve(conv_service):
    await conv_service.save_message("s1", "user", "Hello")
    await conv_service.save_message("s1", "assistant", "Hi there")
    history = await conv_service.get_history("s1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1] == {"role": "assistant", "content": "Hi there"}


async def test_cache_miss_falls_back_to_postgres(conv_service, redis_client):
    await conv_service.save_message("s2", "user", "persist me")
    await redis_client.delete("conversation:s2")
    history = await conv_service.get_history("s2")
    assert len(history) == 1
    assert history[0]["content"] == "persist me"
