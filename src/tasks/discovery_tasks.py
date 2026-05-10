"""
Discovery tasks — run weekly via Celery Beat.

  1. discover_new_channels   — Searches YouTube for similar Spanish investment
                               channels, evaluates relevance with LLM, notifies
                               via Telegram with actionable suggestions.

  2. propose_schema_updates  — Reads recent transcripts, asks LLM to propose
                               new SQL columns/tables that would capture useful
                               information not currently stored, then sends the
                               proposal as a Telegram message.
"""
from __future__ import annotations

import json
from datetime import datetime

import yt_dlp
from loguru import logger
from sqlalchemy import select, desc

from src.celery_app import celery_app
from src.config import settings
from src.llm_client import llm
from src.models import Channel, Transcript
from src.storage.mariadb_client import get_sync_db

# ─────────────────────────────────────────────────────────────────────────────
# Shared Telegram helper
# ─────────────────────────────────────────────────────────────────────────────

def _telegram(text: str) -> None:
    """Fire-and-forget Telegram message (sync, best-effort)."""
    import httpx
    token = settings.telegram_bot_token
    chat_id = settings.telegram_chat_id
    if not token or not chat_id:
        logger.warning("[telegram] No token/chat_id configured — skipping notify")
        return
    try:
        httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as exc:
        logger.warning(f"[telegram] Failed to send message: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Channel Discovery
# ─────────────────────────────────────────────────────────────────────────────

_SEARCH_QUERIES = [
    "análisis bolsa acciones español canal inversión",
    "trading acciones bolsa españa youtube",
    "inversión bolsa española canal youtube análisis",
    "predicciones bolsa acciones español",
    "value investing acciones español youtube",
]

_DISCOVERY_PROMPT = """\
You are evaluating YouTube channels found by search to decide if they should be
added to a monitoring system that tracks Spanish-language stock market predictions.

Existing monitored channels (already in the system):
{existing_channels}

Candidate channels found by search:
{candidates}

For each candidate, answer:
1. Is it a Spanish-language channel focused on stock market analysis/investing? (yes/no)
2. Does it make concrete stock predictions or recommendations? (yes/no)
3. Is it NOT already in the existing list? (yes/no)
4. Overall: RECOMMEND or SKIP

Return a JSON array like:
[
  {{
    "channel_name": "...",
    "channel_url": "...",
    "verdict": "RECOMMEND" | "SKIP",
    "reason": "one sentence"
  }}
]
Only include entries with verdict=RECOMMEND. Return [] if none qualify.
"""


def _search_candidates() -> list[dict]:
    """Search YouTube for Spanish investment channel candidates via yt-dlp."""
    seen_urls: set[str] = set()
    candidates: list[dict] = []

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }

    for query in _SEARCH_QUERIES:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch15:{query}", download=False)
                entries = (info or {}).get("entries") or []
                for entry in entries:
                    ch_url = entry.get("channel_url") or entry.get("uploader_url", "")
                    ch_name = entry.get("channel") or entry.get("uploader", "")
                    if ch_url and ch_url not in seen_urls:
                        seen_urls.add(ch_url)
                        candidates.append({"name": ch_name, "url": ch_url})
        except Exception as exc:
            logger.warning(f"[discovery] Search failed for '{query}': {exc}")

    return candidates


@celery_app.task(
    name="src.tasks.discovery_tasks.discover_new_channels",
    bind=True,
    queue="default",
)
def discover_new_channels(self) -> dict:
    """
    Weekly: search YouTube for Spanish investment channels not yet monitored.
    Uses LLM to filter candidates, then sends Telegram notification.
    """
    logger.info("[discovery] Starting channel discovery")

    # Load existing channels from DB
    with get_sync_db() as db:
        existing = [
            {"name": c.name, "url": c.url}
            for c in db.scalars(select(Channel)).all()
        ]
    existing_names = {e["name"] for e in existing}

    candidates = _search_candidates()
    # Remove already-known channels
    candidates = [c for c in candidates if c["name"] not in existing_names]

    if not candidates:
        logger.info("[discovery] No new candidates found")
        return {"candidates": 0, "recommended": 0}

    logger.info(f"[discovery] Evaluating {len(candidates)} candidates")

    prompt = _DISCOVERY_PROMPT.format(
        existing_channels="\n".join(f"- {e['name']}: {e['url']}" for e in existing),
        candidates="\n".join(f"- {c['name']}: {c['url']}" for c in candidates),
    )

    try:
        raw = llm.chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        # Strip markdown fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        recommendations: list[dict] = json.loads(raw)
    except Exception as exc:
        logger.error(f"[discovery] LLM evaluation failed: {exc}")
        return {"candidates": len(candidates), "recommended": 0, "error": str(exc)}

    if not recommendations:
        logger.info("[discovery] LLM found no qualifying channels")
        _telegram("🔍 *Canal Discovery* — No se encontraron nuevos canales relevantes esta semana.")
        return {"candidates": len(candidates), "recommended": 0}

    # Build Telegram message
    lines = [f"🔍 *Nuevos canales sugeridos para añadir* ({datetime.utcnow().strftime('%Y-%m-%d')})\n"]
    for r in recommendations:
        lines.append(f"📺 *{r.get('channel_name', '?')}*")
        lines.append(f"   URL: {r.get('channel_url', '?')}")
        lines.append(f"   Motivo: {r.get('reason', '')}")
        lines.append("")
    lines.append("Para añadir: `POST /api/v1/channels` con el campo `url`.")
    _telegram("\n".join(lines))

    logger.info(f"[discovery] Recommended {len(recommendations)} new channels")
    return {"candidates": len(candidates), "recommended": len(recommendations)}


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Schema Evolution Agent
# ─────────────────────────────────────────────────────────────────────────────

_CURRENT_SCHEMA_SUMMARY = """\
Current tables and their most relevant columns:

videos: id, channel_id, youtube_video_id, title, description, published_at,
        duration_seconds, transcript_source, processing_status

transcripts: id, video_id, full_text, language, word_count

ticker_mentions: id, video_id, dim_ticker_id, ticker_symbol, company_name,
                 exchange, sentiment(bullish/bearish/neutral), context_snippet,
                 confidence, mention_count

predictions: id, video_id, channel_id, ticker_symbol, prediction_type(buy/sell/hold/watch),
             recommendation(buy/accumulate/hold/reduce/sell/avoid),
             predicted_direction(up/down/neutral), is_long_term, price_at_prediction,
             entry_price, target_price, stop_loss, confidence_score,
             analyst_reasoning, timeframe_days, prediction_date, is_evaluated

ground_truth: id, prediction_id, price_at_evaluation, actual_change_pct,
              actual_direction, is_accurate, evaluation_date, notes

dim_tickers: id, ticker_symbol, company_name, exchange, sector, industry,
             currency, country, is_index, is_etf, is_crypto

channels: id, youtube_channel_id, name, url, is_active, last_checked_at

channel_accuracy: id, channel_id, total_predictions, accurate_predictions,
                  accuracy_pct, avg_return_pct

agent_reports: id, report_type, title, content_markdown, tickers_analyzed,
               video_id, channel_id, telegram_sent, telegram_message_id, telegram_chat_id
"""

_SCHEMA_PROMPT = """\
You are a data engineer analyzing a stock-prediction tracking system.

Current database schema:
{schema}

Recent transcript excerpts (last {n} videos processed):
{transcripts}

Based on these transcripts, identify information that analysts commonly discuss
that is NOT captured by the current schema. Propose 3-7 specific new columns
or tables that would meaningfully improve data quality and analysis capabilities.

For each proposal:
- Table to modify or create
- Column name and SQL type
- Why it adds value (one sentence)
- Example values from the transcripts

Format your response as a clear, numbered Markdown list. Be specific and
practical — no generic suggestions.
"""


@celery_app.task(
    name="src.tasks.discovery_tasks.propose_schema_updates",
    bind=True,
    queue="default",
)
def propose_schema_updates(self) -> dict:
    """
    Weekly: analyze recent transcripts and propose new schema columns/tables
    that would capture useful data not currently stored.
    Sends proposal via Telegram.
    """
    logger.info("[schema-evo] Starting schema evolution analysis")

    # Fetch the last 15 completed transcripts
    with get_sync_db() as db:
        rows = db.execute(
            select(Transcript.full_text)
            .join(Transcript.video)  # type: ignore[arg-type]
            .order_by(desc(Transcript.created_at))
            .limit(15)
        ).scalars().all()

    if not rows:
        logger.info("[schema-evo] No transcripts available")
        return {"transcripts_analyzed": 0}

    # Truncate each transcript to ~800 chars to fit in context
    excerpts = []
    for i, text in enumerate(rows, 1):
        excerpt = (text or "")[:800].replace("\n", " ").strip()
        if excerpt:
            excerpts.append(f"[Transcript {i}]: {excerpt}…")

    combined = "\n\n".join(excerpts)
    prompt = _SCHEMA_PROMPT.format(
        schema=_CURRENT_SCHEMA_SUMMARY,
        n=len(rows),
        transcripts=combined,
    )

    try:
        proposal = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert data engineer proposing database schema improvements. "
                        "Be concise and actionable. Use Markdown formatting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.error(f"[schema-evo] LLM failed: {exc}")
        return {"transcripts_analyzed": len(rows), "error": str(exc)}

    header = (
        f"🗄️ *Propuesta de evolución de esquema* — {datetime.utcnow().strftime('%Y-%m-%d')}\n"
        f"_(basada en {len(rows)} transcripciones recientes)_\n\n"
    )
    # Telegram has 4096-char limit; truncate if needed
    message = (header + proposal)[:4000]
    _telegram(message)

    logger.info(f"[schema-evo] Proposal sent ({len(proposal)} chars)")
    return {"transcripts_analyzed": len(rows), "proposal_length": len(proposal)}
