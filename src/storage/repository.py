"""
Repository — unified data access layer combining MariaDB and ChromaDB.
All business logic that touches multiple data sources lives here.
"""
from datetime import date, datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from src.models import (
    AgentReport,
    Channel,
    ChannelAccuracy,
    DimTicker,
    GroundTruth,
    Prediction,
    ReportType,
    TickerMention,
    Transcript,
    Video,
    ProcessingStatus,
    TranscriptSource,
)
from src.models.base import generate_uuid  # noqa: F401
from src.storage.mariadb_client import get_sync_db


# ── Channel operations ────────────────────────────────────────────────────────

def create_channel(
    db: Session,
    youtube_channel_id: str,
    name: str,
    url: str,
    description: str | None = None,
    thumbnail_url: str | None = None,
) -> Channel:
    channel = Channel(
        youtube_channel_id=youtube_channel_id,
        name=name,
        url=url,
        description=description,
        thumbnail_url=thumbnail_url,
    )
    db.add(channel)
    db.flush()
    return channel


def get_channel_by_yt_id(db: Session, youtube_channel_id: str) -> Channel | None:
    return db.scalar(select(Channel).where(Channel.youtube_channel_id == youtube_channel_id))


def get_all_active_channels(db: Session) -> list[Channel]:
    return list(db.scalars(select(Channel).where(Channel.is_active == True)))  # noqa: E712


def update_channel_last_checked(db: Session, channel_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        update(Channel).where(Channel.id == channel_id).values(last_checked_at=now)
    )


# ── Video operations ──────────────────────────────────────────────────────────

def create_video(db: Session, channel_id: str, **kwargs) -> Video:
    video = Video(channel_id=channel_id, **kwargs)
    db.add(video)
    db.flush()
    return video


def get_video_by_yt_id(db: Session, youtube_video_id: str) -> Video | None:
    return db.scalar(select(Video).where(Video.youtube_video_id == youtube_video_id))


def get_video_by_id(db: Session, video_id: str) -> Video | None:
    return db.scalar(select(Video).where(Video.id == video_id))


def update_video_status(
    db: Session,
    video_id: str,
    status: ProcessingStatus,
    error_message: str | None = None,
) -> None:
    values: dict[str, Any] = {"processing_status": status}
    if error_message is not None:
        values["error_message"] = error_message
    db.execute(update(Video).where(Video.id == video_id).values(**values))


def get_recent_videos(db: Session, limit: int = 20) -> list[Video]:
    return list(
        db.scalars(
            select(Video).order_by(Video.created_at.desc()).limit(limit)
        )
    )


# ── Transcript operations ─────────────────────────────────────────────────────

def save_transcript(
    db: Session,
    video_id: str,
    raw_text: str,
    source: TranscriptSource,
    cleaned_text: str | None = None,
    language: str = "es",
    word_count: int | None = None,
) -> Transcript:
    existing = db.execute(
        select(Transcript).where(Transcript.video_id == video_id)
    ).scalar_one_or_none()

    if existing:
        existing.raw_text = raw_text
        existing.language = language
        if cleaned_text is not None:
            existing.cleaned_text = cleaned_text
        if word_count is not None:
            existing.word_count = word_count
        transcript = existing
    else:
        transcript = Transcript(
            video_id=video_id,
            language=language,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            word_count=word_count,
        )
        db.add(transcript)

    db.execute(
        update(Video).where(Video.id == video_id).values(transcript_source=source)
    )
    db.flush()
    return transcript


def update_transcript_chunks(db: Session, video_id: str, chunk_count: int) -> None:
    db.execute(
        update(Transcript)
        .where(Transcript.video_id == video_id)
        .values(chunk_count=chunk_count)
    )


# ── DimTicker operations ──────────────────────────────────────────────────────

def upsert_dim_ticker(db: Session, ticker_data: dict[str, Any]) -> DimTicker:
    """
    Insert or update a ticker in the dim_tickers dimension table.
    Keyed on ticker_symbol (unique). Returns the DimTicker row.
    """
    symbol = ticker_data.get("ticker_symbol", "").upper().strip()
    if not symbol:
        raise ValueError("ticker_symbol is required")

    existing = db.scalar(select(DimTicker).where(DimTicker.ticker_symbol == symbol))
    if existing:
        # Update enrichment fields if new info is provided
        if ticker_data.get("company_name") and not existing.company_name:
            existing.company_name = ticker_data["company_name"]
        if ticker_data.get("exchange") and not existing.exchange:
            existing.exchange = ticker_data["exchange"]
        if ticker_data.get("sector") and not existing.sector:
            existing.sector = ticker_data["sector"]
        if ticker_data.get("industry") and not existing.industry:
            existing.industry = ticker_data["industry"]
        if ticker_data.get("currency") and not existing.currency:
            existing.currency = ticker_data["currency"]
        if ticker_data.get("country") and not existing.country:
            existing.country = ticker_data["country"]
        db.flush()
        return existing

    dim = DimTicker(
        ticker_symbol=symbol,
        company_name=ticker_data.get("company_name"),
        exchange=ticker_data.get("exchange"),
        sector=ticker_data.get("sector"),
        industry=ticker_data.get("industry"),
        currency=ticker_data.get("currency"),
        country=ticker_data.get("country"),
        is_index=bool(ticker_data.get("is_index", False)),
        is_etf=bool(ticker_data.get("is_etf", False)),
        is_crypto=bool(ticker_data.get("is_crypto", False)),
    )
    db.add(dim)
    db.flush()
    return dim


# ── Ticker mention operations ─────────────────────────────────────────────────

def save_ticker_mentions(
    db: Session, video_id: str, mentions: list[dict[str, Any]]
) -> list[TickerMention]:
    objects = []
    for m in mentions:
        symbol = m.get("ticker_symbol", "").upper().strip()
        if not symbol:
            continue

        # Upsert the dimension row first
        try:
            dim = upsert_dim_ticker(db, m)
            dim_ticker_id = dim.id
        except Exception:
            dim_ticker_id = None

        obj = TickerMention(
            video_id=video_id,
            dim_ticker_id=dim_ticker_id,
            ticker_symbol=symbol,
            company_name=m.get("company_name"),
            exchange=m.get("exchange"),
            sentiment=m.get("sentiment", "neutral"),
            mention_count=max(1, int(m.get("mention_count", 1))),
            context_snippet=m.get("context_snippet"),
            confidence=m.get("confidence"),
        )
        db.add(obj)
        objects.append(obj)
    db.flush()
    return objects


def get_ticker_mentions(
    db: Session, ticker_symbol: str, limit: int = 50
) -> list[TickerMention]:
    return list(
        db.scalars(
            select(TickerMention)
            .where(TickerMention.ticker_symbol == ticker_symbol.upper())
            .order_by(TickerMention.created_at.desc())
            .limit(limit)
        )
    )


# ── Prediction operations ─────────────────────────────────────────────────────

def save_predictions(
    db: Session,
    video_id: str,
    channel_id: str,
    predictions: list[dict[str, Any]],
    price_lookup: dict[str, float],
) -> list[Prediction]:
    objects = []
    today = date.today().isoformat()
    for p in predictions:
        ticker = p.get("ticker_symbol", "").upper().strip()
        if not ticker:
            continue

        # Validate enums — fall back to safe defaults
        pred_type = p.get("prediction_type", "watch")
        if pred_type not in ("buy", "sell", "hold", "watch"):
            pred_type = "watch"

        pred_dir = p.get("predicted_direction", "neutral")
        if pred_dir not in ("up", "down", "neutral"):
            pred_dir = "neutral"

        rec = p.get("recommendation")
        if rec not in ("buy", "accumulate", "hold", "reduce", "sell", "avoid", None):
            rec = None

        obj = Prediction(
            video_id=video_id,
            channel_id=channel_id,
            ticker_symbol=ticker,
            prediction_type=pred_type,
            recommendation=rec,
            predicted_direction=pred_dir,
            is_long_term=p.get("is_long_term"),
            price_at_prediction=p.get("price_at_prediction") or price_lookup.get(ticker),
            entry_price=p.get("entry_price"),
            target_price=p.get("target_price"),
            stop_loss=p.get("stop_loss"),
            timeframe_days=p.get("timeframe_days"),
            confidence_score=p.get("confidence_score"),
            analyst_reasoning=p.get("analyst_reasoning"),
            context_snippet=p.get("context_snippet"),
            prediction_date=today,
        )
        db.add(obj)
        objects.append(obj)
    db.flush()
    return objects


def get_pending_predictions(db: Session) -> list[Prediction]:
    return list(
        db.scalars(
            select(Prediction)
            .where(Prediction.is_evaluated == False)  # noqa: E712
            .order_by(Prediction.prediction_date)
        )
    )


def save_ground_truth(
    db: Session,
    prediction_id: str,
    price_at_evaluation: float,
    actual_change_pct: float,
    actual_direction: str,
    is_accurate: bool,
) -> GroundTruth:
    gt = GroundTruth(
        prediction_id=prediction_id,
        price_at_evaluation=price_at_evaluation,
        actual_change_pct=actual_change_pct,
        actual_direction=actual_direction,
        is_accurate=is_accurate,
        evaluation_date=date.today().isoformat(),
    )
    db.add(gt)
    db.execute(
        update(Prediction)
        .where(Prediction.id == prediction_id)
        .values(is_evaluated=True)
    )
    db.flush()
    return gt


def refresh_channel_accuracy(db: Session, channel_id: str) -> None:
    total = db.scalar(
        select(func.count(Prediction.id))
        .where(Prediction.channel_id == channel_id, Prediction.is_evaluated == True)  # noqa: E712
    ) or 0
    accurate = db.scalar(
        select(func.count(GroundTruth.id))
        .join(Prediction)
        .where(Prediction.channel_id == channel_id, GroundTruth.is_accurate == True)  # noqa: E712
    ) or 0

    accuracy_pct = round((accurate / total) * 100, 2) if total > 0 else None

    existing = db.scalar(
        select(ChannelAccuracy).where(ChannelAccuracy.channel_id == channel_id)
    )
    if existing:
        existing.total_predictions = total
        existing.accurate_predictions = accurate
        existing.accuracy_pct = accuracy_pct
    else:
        db.add(ChannelAccuracy(
            channel_id=channel_id,
            total_predictions=total,
            accurate_predictions=accurate,
            accuracy_pct=accuracy_pct,
        ))
    db.flush()


# ── Report operations ─────────────────────────────────────────────────────────

def save_report(
    db: Session,
    report_type: ReportType,
    title: str,
    content_markdown: str,
    tickers_analyzed: list[str] | None = None,
    video_id: str | None = None,
    channel_id: str | None = None,
    telegram_chat_id: str | None = None,
) -> AgentReport:
    report = AgentReport(
        report_type=report_type,
        title=title,
        content_markdown=content_markdown,
        tickers_analyzed=tickers_analyzed,
        video_id=video_id,
        channel_id=channel_id,
        telegram_chat_id=telegram_chat_id,
    )
    db.add(report)
    db.flush()
    return report


def mark_report_sent(db: Session, report_id: str, telegram_message_id: int) -> None:
    db.execute(
        update(AgentReport)
        .where(AgentReport.id == report_id)
        .values(telegram_sent=True, telegram_message_id=telegram_message_id)
    )


def get_channel_accuracy_all(db: Session) -> list[ChannelAccuracy]:
    return list(db.scalars(select(ChannelAccuracy)))
