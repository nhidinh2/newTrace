# NewsTrace — CPU-only image. No API key required to run the fixture demo.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NEWSTRACE_DATABASE_URL=sqlite:////data/newstrace.db \
    NEWSTRACE_EMBEDDING_BACKEND=auto \
    LLM_PROVIDER=none

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Install the package plus the UI/report extras. The `embeddings` extra pulls in
# torch; leave it out and NewsTrace falls back to the deterministic hashing
# embedder, which keeps the image small and the demo working offline.
RUN pip install --upgrade pip && pip install ".[ui,reports]"

COPY alembic.ini Makefile ./
COPY migrations ./migrations
COPY configs ./configs
COPY data ./data
COPY app ./app
COPY scripts ./scripts
COPY docs ./docs

RUN mkdir -p /data /app/artifacts

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Apply migrations, load the committed fixtures, then serve the API.
CMD ["sh", "-c", "alembic upgrade head && python scripts/ingest_demo.py --source fixtures && uvicorn newstrace.api.app:app --host 0.0.0.0 --port 8000"]
