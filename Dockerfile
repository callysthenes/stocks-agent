# ── Base stage ────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv==0.5.11

WORKDIR /app

# Install dependencies (no GPU extras)
COPY pyproject.toml .
RUN uv pip install --system --no-cache \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "pydantic>=2.9.0" \
    "pydantic-settings>=2.6.0" \
    "python-multipart>=0.0.12" \
    "sqlalchemy>=2.0.36" \
    "alembic>=1.14.0" \
    "pymysql>=1.1.1" \
    "aiomysql>=0.2.0" \
    "cryptography>=43.0.3" \
    "celery[redis]>=5.4.0" \
    "redis>=5.2.1" \
    "yt-dlp>=2024.11.4" \
    "youtube-transcript-api>=0.6.3" \
    "chromadb>=0.5.23" \
    "llama-index-core>=0.12.0" \
    "llama-index-vector-stores-chroma>=0.3.0" \
    "llama-index-embeddings-huggingface>=0.4.0" \
    "llama-index-llms-openai>=0.3.0" \
    "langchain>=0.3.7" \
    "langchain-openai>=0.2.10" \
    "langgraph>=0.2.55" \
    "langchain-community>=0.3.7" \
    "yfinance>=0.2.50" \
    "pandas-ta>=0.3.14b0" \
    "pandas>=2.2.3" \
    "numpy>=1.26.4" \
    "langfuse>=2.0.0" \
    "python-telegram-bot>=21.7" \
    "httpx>=0.27.2" \
    "aiofiles>=24.1.0" \
    "loguru>=0.7.3" \
    "tenacity>=9.0.0" \
    "python-dateutil>=2.9.0" \
    "pytz>=2024.2" \
    "ujson>=5.10.0"

COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY scripts/ ./scripts/

# ── API stage ─────────────────────────────────────────────────────────────────
FROM base AS api

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
