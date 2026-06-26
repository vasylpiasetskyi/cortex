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
