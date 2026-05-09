"""
Ingestion pipeline — orchestrates the full processing chain for a video:
clean → chunk → embed → extract metadata → store everything.
"""
from datetime import date
from typing import Any

import yfinance as yf
from loguru import logger

from src.config import settings
from src.ingestion.chunker import chunk_text
from src.ingestion.embedding_generator import embed_texts
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.transcript_cleaner import clean_transcript, count_words
from src.models import ProcessingStatus, TranscriptSource
from src.storage import chroma_client, repository
from src.storage.mariadb_client import get_sync_db

_extractor: MetadataExtractor | None = None


def get_extractor() -> MetadataExtractor:
    global _extractor
    if _extractor is None:
        _extractor = MetadataExtractor()
    return _extractor


def run_ingestion_pipeline(
    video_id: str,
    raw_transcript: str,
    transcript_source: TranscriptSource,
) -> dict[str, Any]:
    """
    Full ingestion pipeline for a single video.

    Steps:
        1. Clean transcript
        2. Extract ticker mentions (DeepSeek)
        3. Extract predictions (DeepSeek)
        4. Chunk transcript
        5. Generate embeddings (bge-m3)
        6. Store everything in MariaDB + ChromaDB

    Returns:
        Summary dict with counts.
    """
    logger.info(f"Starting ingestion pipeline for video {video_id}")

    with get_sync_db() as db:
        # Get video + channel info
        video = repository.get_video_by_id(db, video_id)
        if not video:
            raise ValueError(f"Video {video_id} not found in DB")
        channel_id = video.channel_id
        channel_name = video.channel.name if video.channel else "Unknown"

        try:
            # ── Step 1: Update status ─────────────────────────────────────────
            repository.update_video_status(db, video_id, ProcessingStatus.embedding)

            # ── Step 2: Clean transcript ──────────────────────────────────────
            cleaned = clean_transcript(raw_transcript)
            word_count = count_words(cleaned)

            # Save transcript to MariaDB
            repository.save_transcript(
                db,
                video_id=video_id,
                raw_text=raw_transcript,
                source=transcript_source,
                cleaned_text=cleaned,
                word_count=word_count,
                language="es",
            )

            # ── Step 3: Extract tickers ───────────────────────────────────────
            extractor = get_extractor()
            ticker_dicts = extractor.extract_tickers(cleaned)
            logger.info(f"Extracted {len(ticker_dicts)} tickers from video {video_id}")

            repository.save_ticker_mentions(db, video_id, ticker_dicts)

            # ── Step 4: Extract predictions ───────────────────────────────────
            prediction_dicts = extractor.extract_predictions(cleaned, ticker_dicts)
            logger.info(f"Extracted {len(prediction_dicts)} predictions from video {video_id}")

            # Fetch current prices for predictions
            price_lookup: dict[str, float] = {}
            if prediction_dicts:
                price_lookup = _fetch_current_prices(
                    [p["ticker_symbol"] for p in prediction_dicts if p.get("ticker_symbol")]
                )

            repository.save_predictions(db, video_id, channel_id, prediction_dicts, price_lookup)

            # ── Step 5: Chunk transcript ──────────────────────────────────────
            chunks = chunk_text(cleaned)
            logger.info(f"Created {len(chunks)} chunks from video {video_id}")

            repository.update_transcript_chunks(db, video_id, len(chunks))

            # ── Step 6: Generate embeddings ───────────────────────────────────
            chunk_texts = [c.text for c in chunks]
            embeddings = embed_texts(chunk_texts)

            # ── Step 7: Store in ChromaDB ─────────────────────────────────────
            chunk_ids = [f"{video_id}_{c.chunk_index}" for c in chunks]
            metadatas = [
                {
                    "video_id": video_id,
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "video_title": video.title,
                    "published_at": video.published_at or "",
                    "chunk_index": c.chunk_index,
                    "token_estimate": c.token_estimate,
                }
                for c in chunks
            ]

            chroma_client.upsert_chunks(chunk_ids, chunk_texts, embeddings, metadatas)

            # ── Step 8: Mark as completed ────────────────────────────────────
            repository.update_video_status(db, video_id, ProcessingStatus.completed)
            logger.info(f"Ingestion completed for video {video_id}")

            return {
                "video_id": video_id,
                "word_count": word_count,
                "chunk_count": len(chunks),
                "ticker_count": len(ticker_dicts),
                "prediction_count": len(prediction_dicts),
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Ingestion pipeline failed for video {video_id}: {e}")
            repository.update_video_status(
                db, video_id, ProcessingStatus.failed, error_message=str(e)
            )
            raise


def _fetch_current_prices(ticker_symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for a list of tickers (best effort)."""
    prices: dict[str, float] = {}
    unique_tickers = list(set(t.upper() for t in ticker_symbols if t))

    for ticker in unique_tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            price = info.last_price or info.previous_close
            if price:
                prices[ticker] = float(price)
        except Exception:
            pass  # Price lookup is best-effort; non-critical

    return prices
