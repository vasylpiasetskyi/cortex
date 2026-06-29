from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class Person(BaseModel):
    name: str
    age: int


class ExtractRequest(BaseModel):
    text: str


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    file_size: int
    error_msg: str | None = None
    embedding_model: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    status: str
    file_size: int


class SourceOut(BaseModel):
    page: int
    chunk_index: int
    score: float


class AskRequest(BaseModel):
    session_id: str
    question: str
    document_id: int | None = None
    strategy: Literal["baseline", "sentence_window", "auto_merging"] = "baseline"


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    strategy: str
