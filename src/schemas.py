"""Pydantic-схемы запросов и ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=15000,
        description="Фрагмент текста или запрос пользователя (до 15 000 символов)",
    )
    k: int = Field(default=3, ge=1, le=10, description="Параметр для top-k авторов (опционально)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class AuthorScore(BaseModel):
    author: str
    probability: float


class QueryResponse(BaseModel):
    predicted_author: str
    confidence: float
    model_name: str
    generated_text: str


class ClassifyResponse(BaseModel):
    predicted_author: str
    confidence: float
    model_name: str
    top_authors: list[AuthorScore]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
