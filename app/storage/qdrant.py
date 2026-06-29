from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "cortex_chunks"


def get_qdrant_client(url: str) -> AsyncQdrantClient:
    return AsyncQdrantClient(url=url)


async def init_collection(client: AsyncQdrantClient, vector_size: int) -> None:
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if COLLECTION_NAME in names:
        info = await client.get_collection(COLLECTION_NAME)
        existing_size = info.config.params.vectors.size
        if existing_size != vector_size:
            logger.error(
                f"Qdrant collection vector size mismatch "
                f"(existing={existing_size}, expected={vector_size}). "
                f"Run `make reset-vectors` to recreate the collection. "
                f"All documents will need reindexing."
            )
        return
    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
