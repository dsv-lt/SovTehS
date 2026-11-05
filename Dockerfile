FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Шаг 1: Установка зависимостей с кэшированием слоя
# Сначала ставится легковесная CPU-сборка PyTorch (экономия ~2.5 ГБ относительно CUDA-колеса)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoTokenizer.from_pretrained('ai-forever/ruBert-base'); AutoModelForSequenceClassification.from_pretrained('ai-forever/ruBert-base')"

# Создание непривилегированного пользователя для безопасного запуска
RUN useradd -u 1000 -m appuser && chown -R appuser:appuser /app

# Шаг 2: Копирование исходного кода и обученных артефактов
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

USER appuser

EXPOSE 8000

# Зонд жизнеспособности контейнера
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Шаг 3: Точка входа
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
