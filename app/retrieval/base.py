from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    page: int
    chunk_index: int
    score: float
    text: str = field(default="")


class RetrieverStrategy(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query_vector: list[float],
        session_id: str,
        document_id: int | None,
        top_k: int,
    ) -> list[RetrievedChunk]: ...
