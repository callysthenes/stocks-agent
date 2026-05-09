"""
FastAPI application entry point.
Provides REST API for channel management, video queries, agent interaction,
and system health checks.
"""
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    logger.info(f"Starting StocksAgent API [{settings.environment}]")
    # Initialize DB tables on startup (idempotent)
    from src.storage.mariadb_client import init_db
    await init_db()
    yield
    logger.info("Shutting down StocksAgent API")


app = FastAPI(
    title="StocksAgent API",
    description="AI-powered stock analysis from Spanish YouTube investment channels",
    version="0.1.0",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan,
)

# CORS — restrict to internal network in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else ["http://dashboard:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} [{elapsed:.0f}ms]"
    )
    return response


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok", "version": "0.1.0", "env": settings.environment}


# ── Routers ───────────────────────────────────────────────────────────────────

from src.api import channels, videos, tickers, agent, reports, predictions  # noqa: E402

app.include_router(channels.router, prefix="/api/v1/channels", tags=["channels"])
app.include_router(videos.router, prefix="/api/v1/videos", tags=["videos"])
app.include_router(tickers.router, prefix="/api/v1/tickers", tags=["tickers"])
app.include_router(agent.router, prefix="/api/v1/agent", tags=["agent"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
