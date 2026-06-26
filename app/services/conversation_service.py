import json

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.db import Message

_CACHE_KEY = "conversation:{session_id}"
_CACHE_TTL = 86400  # 24 hours
_MAX_MESSAGES = 20


class ConversationService:
    def __init__(self, redis: Redis, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.redis = redis
        self.session_factory = session_factory

    async def get_history(self, session_id: str) -> list[dict]:
        key = _CACHE_KEY.format(session_id=session_id)
        cached = await self.redis.lrange(key, 0, -1)
        if cached:
            return [json.loads(m) for m in cached]
        return await self._load_from_postgres(session_id)

    async def save_message(self, session_id: str, role: str, content: str) -> None:
        async with self.session_factory() as session:
            session.add(Message(session_id=session_id, role=role, content=content))
            await session.commit()
        key = _CACHE_KEY.format(session_id=session_id)
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.rpush(key, json.dumps({"role": role, "content": content}))
            pipe.ltrim(key, -_MAX_MESSAGES, -1)
            pipe.expire(key, _CACHE_TTL)
            await pipe.execute()

    async def _load_from_postgres(self, session_id: str) -> list[dict]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at)
            )
            rows = result.scalars().all()
        history = [{"role": m.role, "content": m.content} for m in rows]
        if history:
            key = _CACHE_KEY.format(session_id=session_id)
            async with self.redis.pipeline(transaction=False) as pipe:
                pipe.delete(key)
                for msg in history:
                    pipe.rpush(key, json.dumps(msg))
                pipe.ltrim(key, -_MAX_MESSAGES, -1)
                pipe.expire(key, _CACHE_TTL)
                await pipe.execute()
        return history
