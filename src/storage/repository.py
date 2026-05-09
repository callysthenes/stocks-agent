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
            select(Video)
            .order_by(Video.created_at.desc())
            .limit(limit)
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
    transcript = Transcript(
        video_id=video_id,
        language=language,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
        word_count=word_count,
    )
    db.add(transcript)
    db.execute(
        update(Video)
        .where(Video.id == video_id)
        .values(transcript_source=source)
    )
    db.flush()
    return transcript


def update_transcript_chunks(db: Session, video_id: str, chunk_count: int) -> None:
    db.execute(
        update(Transcript)
        .where(Transcript.video_id == video_id)
        .values(chunk_count=chunk_count)
    )


# ── Ticker mention operations ─────────────────────────────────────────────────

def save_ticker_mentions(
    db: Session, video_id: str, mentions: list[dict[str, Any]]
) -> list[TickerMention]:
    objects = []
    for m in mentions:
        obj = TickerMention(
            video_id=video_id,
            ticker_symbol=m.get("ticker_symbol", "").upper(),
            company_name=m.get("company_name"),
            exchange=m.get("exchange"),
            sentiment=m.get("sentiment", "neutral"),
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
        ticker = p.get("ticker_symbol", "").upper()
        obj = Prediction(
            video_id=video_id,
            channel_id=channel_id,
            ticker_symbol=ticker,
            prediction_type=p.get("prediction_type", "watch"),
            predicted_direction=p.get("predicted_direction", "neutral"),
            target_price=p.get("target_price"),
            timeframe_days=p.get("timeframe_days"),
            price_at_prediction=price_lookup.get(ticker),
            prediction_date=today,
            context_snippet=p.get("context_snippet"),
        )
        db.add(obj)
        objects.append(obj)
    db.flush()
    return objects


def get_pending_predictions(db: Session) -> list[Prediction]:
    """Get predictions not yet evaluated whose timeframe may have elapsed."""
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
    """Recalculate and upsert channel accuracy summary."""
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
) -> AgentReport:
    report = AgentReport(
        report_type=report_type,
        title=title,
        content_markdown=content_markdown,
        tickers_analyzed=tickers_analyzed,
        video_id=video_id,
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
