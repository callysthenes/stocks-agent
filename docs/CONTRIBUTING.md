# Contributing to StocksAgent

## Setup

### Prerequisites
- Docker + Docker Compose v2
- NVIDIA GPU(s) with nvidia-container-toolkit (for worker)
- Git

### Quick Start

```bash
# Clone the repo
git clone https://github.com/your-org/stocks-agent.git
cd stocks-agent

# Copy environment config
cp .env.example .env
# Edit .env and fill in your secrets

# Start all services
docker compose up -d

# Check logs
docker compose logs -f api
docker compose logs -f worker

# Run database migrations
docker compose exec api alembic upgrade head

# Seed initial channels
docker compose exec api python scripts/seed_channels.py

# Access services
# API docs:   http://localhost:8000/docs
# Dashboard:  http://localhost:8501
# Langfuse:   http://localhost:3000
```

### Development Mode (hot-reload)

```bash
# Use the dev override for hot-reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Running Tests

```bash
# Install dev dependencies
pip install uv
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Lint
ruff check src/ dashboard/

# Format
ruff format src/ dashboard/

# Type check
mypy src/
```

## Project Structure

```
stocks_agent/
├── src/                    # Application source
│   ├── config.py           # Pydantic Settings (all env vars)
│   ├── main.py             # FastAPI app
│   ├── celery_app.py       # Celery configuration + beat schedule
│   ├── models/             # SQLAlchemy ORM models
│   ├── storage/            # MariaDB + ChromaDB clients + repository
│   ├── input/              # YouTube fetcher + Whisper transcriber
│   ├── ingestion/          # Cleaning, chunking, embedding, extraction
│   ├── retrieval/          # LlamaIndex RAG pipeline
│   ├── agents/             # LangGraph multi-agent system
│   ├── tasks/              # Celery tasks
│   ├── delivery/           # Telegram bot + report formatter
│   └── observability/      # Langfuse integration
├── dashboard/              # Streamlit dashboard (5 pages)
├── tests/                  # Pytest tests
├── scripts/                # CLI utilities (seed, backfill)
├── alembic/                # Database migrations
├── docker-compose.yml      # Production compose
├── docker-compose.dev.yml  # Development overrides
├── Dockerfile              # API + beat + bot image
├── Dockerfile.worker       # GPU worker image (CUDA 12.4)
└── Dockerfile.dashboard    # Streamlit image
```

## Adding a YouTube Channel

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/channels \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/@ChannelName"}'

# Or via script (add URL to scripts/seed_channels.py first)
docker compose exec api python scripts/seed_channels.py
```

## GPU Configuration

The worker container uses GPU 0 for:
- Whisper `large-v3` transcription (~4.5 GB VRAM)
- `bge-m3` embedding model (~2.2 GB VRAM)

Ollama uses GPU 1 for future local LLM inference.

To change GPU assignments, edit `docker-compose.yml`:
```yaml
worker:
  environment:
    CUDA_VISIBLE_DEVICES: "0"  # Change GPU index here
```

## Environment Variables

See `.env.example` for all required and optional variables.
**Never commit your `.env` file.**

## Disclaimer

StocksAgent is for educational and research purposes only.
All AI-generated analysis is not financial advice.
