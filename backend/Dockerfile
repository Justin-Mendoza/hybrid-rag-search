FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

CMD ["uvicorn", "hybrid_rag_search.main:app", "--app-dir", "/app/src", "--host", "0.0.0.0", "--port", "8000"]
