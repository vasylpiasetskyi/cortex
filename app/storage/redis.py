from redis.asyncio import Redis


def get_redis(url: str) -> Redis:
    return Redis.from_url(url, decode_responses=True)
