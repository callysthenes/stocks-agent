"""
Prediction tracking tools for the Analysis Agent.
CRUD operations for predictions and ground truth evaluation.
"""
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import select

from src.models import Channel, ChannelAccuracy, GroundTruth, Prediction
from src.storage import repository
from src.storage.mariadb_client import get_sync_db


def get_active_predictions_for_ticker(ticker_symbol: str) -> list[dict[str, Any]]:
    """Get all active (not yet evaluated) predictions for a ticker."""
    with get_sync_db() as db:
        preds = db.scalars(
            select(Prediction)
            .where(
                Prediction.ticker_symbol == ticker_symbol.upper(),
                Prediction.is_evaluated == False,  # noqa: E712
            )
            .order_by(Prediction.prediction_date.desc())
        ).all()

        results = []
        for p in preds:
            results.append({
                "id": p.id,
                "ticker_symbol": p.ticker_symbol,
                "channel_id": p.channel_id,
                "prediction_type": p.prediction_type,
                "predicted_direction": p.predicted_direction,
                "target_price": float(p.target_price) if p.target_price else None,
                "timeframe_days": p.timeframe_days,
                "price_at_prediction": float(p.price_at_prediction) if p.price_at_prediction else None,
                "prediction_date": p.prediction_date,
                "context_snippet": p.context_snippet,
            })
        return results


def get_channel_accuracy_summary() -> list[dict[str, Any]]:
    """Get accuracy summary for all channels."""
    with get_sync_db() as db:
        rows = db.scalars(
            select(ChannelAccuracy).join(Channel)
        ).all()

        results = []
        for row in rows:
            channel = db.scalar(select(Channel).where(Channel.id == row.channel_id))
            results.append({
                "channel_id": row.channel_id,
                "channel_name": channel.name if channel else "Unknown",
                "total_predictions": row.total_predictions,
                "accurate_predictions": row.accurate_predictions,
                "accuracy_pct": float(row.accuracy_pct) if row.accuracy_pct else None,
                "avg_return_pct": float(row.avg_return_pct) if row.avg_return_pct else None,
            })
        return sorted(results, key=lambda x: x.get("accuracy_pct") or 0, reverse=True)


def get_prediction_history_for_ticker(
    ticker_symbol: str, limit: int = 10
) -> list[dict[str, Any]]:
    """Get historical predictions (evaluated) for a ticker, ordered by date."""
    with get_sync_db() as db:
        preds = db.scalars(
            select(Prediction)
            .where(
                Prediction.ticker_symbol == ticker_symbol.upper(),
                Prediction.is_evaluated == True,  # noqa: E712
            )
            .order_by(Prediction.prediction_date.desc())
            .limit(limit)
        ).all()

        results = []
        for p in preds:
            gt = p.ground_truth
            channel = p.channel
            results.append({
                "ticker_symbol": p.ticker_symbol,
                "channel_name": channel.name if channel else "Unknown",
                "prediction_type": p.prediction_type,
                "predicted_direction": p.predicted_direction,
                "prediction_date": p.prediction_date,
                "price_at_prediction": float(p.price_at_prediction) if p.price_at_prediction else None,
                "was_accurate": gt.is_accurate if gt else None,
                "actual_change_pct": float(gt.actual_change_pct) if gt and gt.actual_change_pct else None,
                "evaluation_date": gt.evaluation_date if gt else None,
            })
        return results
