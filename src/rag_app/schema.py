from typing import Any
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    source: str | None = None
    title: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SourceDocument(BaseModel):
    id: str
    text: str
    metadata: Metadata = Field(default_factory=Metadata)


class QueryRequest(BaseModel):
    query: str


class SourceResponse(BaseModel):
    id: str
    text: str
    metadata: Metadata
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class IngestRequest(BaseModel):
    documents: list[SourceDocument]
