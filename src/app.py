"""FastAPI-приложение двухстадийного сервиса классификации стиля и генерации текста."""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import joblib
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.classifier import StyleClassifier, load_classifier
from src.config import get_settings
from src.llm_client import LLMClient, LLMError, build_system_prompt
from src.preprocessing import normalize_for_model
from src.schemas import (
    AuthorScore,
    ClassifyResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Единоразовая загрузка моделей в память при старте и освобождение при остановке."""
    settings = get_settings()
    logger.info("Starting Literary Style Assistant (backend=%s)", settings.classifier_backend)

    vectorizer = None
    if settings.vectorizer_path.exists():
        logger.info("Loading TF-IDF vectorizer from %s", settings.vectorizer_path)
        vectorizer = joblib.load(settings.vectorizer_path)

    classifier = load_classifier(settings, vectorizer=vectorizer)
    llm_client = LLMClient(settings)

    app.state.classifier = classifier
    app.state.vectorizer = vectorizer
    app.state.llm_client = llm_client

    logger.info("Application ready to serve requests (classifier=%s)", classifier.name)
    try:
        yield
    finally:
        logger.info("Shutting down service and cleaning up resources")
        await llm_client.aclose()
        classifier.close()
        app.state.classifier = None
        app.state.vectorizer = None
        app.state.llm_client = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="Literary Style Assistant API",
    description="Двухстадийный сервис классификации литературного стиля 64 русских авторов и стилизованной генерации текста.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Возвращает 400 «Пустой вопрос» при пустой строке question, сохраняя 422 для прочих ошибок."""
    for err in exc.errors():
        if err.get("loc") == ("body", "question") and err.get("type") in (
            "string_too_short",
            "value_error.missing",
        ):
            return JSONResponse(status_code=400, content={"detail": "Пустой вопрос"})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _require_classifier(request: Request) -> StyleClassifier:
    classifier: StyleClassifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель классификатора не загружена")
    return classifier


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Проверка готовности сервиса и загрузки локальной модели."""
    loaded = getattr(request.app.state, "classifier", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(payload: QueryRequest, request: Request) -> ClassifyResponse:
    """Локальная классификация стиля автора (стадия 1 без обращения к внешней LLM)."""
    # Нормализация удаляет управляющие символы и невидимые пробелы
    cleaned = normalize_for_model(payload.question)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    classifier = _require_classifier(request)
    try:
        pred = await asyncio.to_thread(classifier.predict, cleaned, payload.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Пустой вопрос") from exc

    return ClassifyResponse(
        predicted_author=pred.author,
        confidence=round(pred.confidence, 4),
        model_name=classifier.name,
        top_authors=[
            AuthorScore(author=author, probability=round(prob, 4)) for author, prob in pred.top_k
        ],
    )


@app.post("/ask", response_model=QueryResponse)
async def ask(payload: QueryRequest, request: Request) -> QueryResponse:
    """Полный конвейер: классификация автора (стадия 1) + генерация в его стиле (стадия 2)."""
    cleaned = normalize_for_model(payload.question)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    classifier = _require_classifier(request)
    llm_client: LLMClient = request.app.state.llm_client

    try:
        pred = await asyncio.to_thread(classifier.predict, cleaned, payload.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Пустой вопрос") from exc

    system_prompt = build_system_prompt(pred.author)

    try:
        generated_text = await llm_client.generate(
            system_prompt=system_prompt,
            user_message=cleaned,
            temperature=payload.temperature,
        )
    except LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return QueryResponse(
        predicted_author=pred.author,
        confidence=round(pred.confidence, 4),
        model_name=classifier.name,
        generated_text=generated_text,
    )
