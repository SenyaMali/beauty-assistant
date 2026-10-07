# Образ API-сервиса. Qdrant — отдельный контейнер, Ollama — на компьютере (см. docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Кэш моделей fastembed — в папку models/, которая монтируется с компьютера:
    # модель эмбеддингов (~2,2 ГБ) скачивается один раз и переживает пересборку образа.
    FASTEMBED_CACHE_PATH=/app/models/fastembed_cache

WORKDIR /app

# Сначала только зависимости: этот слой кэшируется и не пересобирается при правке кода.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

EXPOSE 8000

# При старте: проиндексировать каталог, если индекса ещё нет, затем поднять API.
CMD ["sh", "-c", "python -m app.ingest --if-empty && uvicorn app.api:app --host 0.0.0.0 --port 8000"]
