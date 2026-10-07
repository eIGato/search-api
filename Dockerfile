# syntax=docker/dockerfile:1

FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable


FROM python:3.13-slim
# /app itself is owned by the runtime user (for the model cache); the code stays root-owned.
RUN useradd --system --create-home app && mkdir /app && chown app:app /app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    EMBEDDING_CACHE_DIR=/app/models
USER app
# Bake the embedding model into the image: the container starts fast and needs no network.
RUN python -c "import os; from search_api.embeddings import FastEmbedEmbedder as E; \
E(os.environ['EMBEDDING_MODEL'], os.environ['EMBEDDING_CACHE_DIR'])" && rm -f ./*.ses
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "alembic upgrade head && exec search-api --host 0.0.0.0 --port 8000"]
