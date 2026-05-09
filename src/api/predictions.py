"""FastAPI predictions router."""
from fastapi import APIRouter, Query
from sqlalchemy import desc, select

from src.models import GroundTruth, Prediction
from src.storage.mariadb_client import get_sync_db

router = APIRouter()


@router.get("/")
async def list_predictions(
    limit: int = Query(50, le=200),
    ticker: str | None = None,
    evaluated: bool | None = None,
):
    """List predictions with optional filters."""
    from sqlalchemy import and_
    with get_sync_db() as db:
        query = select(Prediction).order_by(desc(Prediction.created_at)).limit(limit)
        filters = []
        if ticker:
            filters.append(Prediction.ticker_symbol == ticker.upper())
        if evaluated is not None:
            filters.append(Prediction.is_evaluated == evaluated)
        if filters:
            query = query.where(and_(*filters))

        preds = db.scalars(query).all()
        return [
            {
                "id": p.id,
                "ticker_symbol": p.ticker_symbol,
                "channel_id": p.channel_id,
                "prediction_type": p.prediction_type,
                "predicted_direction": p.predicted_direction,
                "target_price": float(p.target_price) if p.target_price else None,
                "timeframe_days": p.timeframe_days,
                "price_at_prediction": float(p.price_at_prediction) if p.price_at_prediction else None,
                "prediction_date": p.prediction_date,
                "is_evaluated": p.is_evaluated,
                "ground_truth": {
                    "is_accurate": p.ground_truth.is_accurate,
                    "actual_change_pct": float(p.ground_truth.actual_change_pct)
                    if p.ground_truth and p.ground_truth.actual_change_pct
                    else None,
                    "actual_direction": p.ground_truth.actual_direction,
                    "evaluation_date": p.ground_truth.evaluation_date,
                }
                if p.ground_truth
                else None,
            }
            for p in preds
        ]
