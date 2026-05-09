"""
Celery analysis tasks — ground truth evaluation, accuracy calculation.
Runs on the default queue (no GPU required).
"""
from datetime import date, datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select

from src.celery_app import celery_app
from src.models import GroundTruth, Prediction
from src.storage import repository
from src.storage.mariadb_client import get_sync_db


@celery_app.task(
    name="src.tasks.analysis_tasks.evaluate_pending_predictions",
    bind=True,
    queue="default",
)
def evaluate_pending_predictions(self) -> dict:
    """
    Evaluate all pending predictions whose timeframe has elapsed.
    Fetches current price, compares with prediction, stores ground truth.
    Triggered daily by Celery Beat at 22:00 UTC.
    """
    logger.info("Starting daily prediction evaluation")

    from src.agents.tools.financial_tools import get_current_price

    with get_sync_db() as db:
        pending = repository.get_pending_predictions(db)

    today = date.today()
    evaluated = 0
    skipped = 0
    errors = 0

    for pred in pending:
        try:
            # Check if the prediction timeframe has elapsed
            if pred.prediction_date and pred.timeframe_days:
                pred_date = datetime.strptime(pred.prediction_date, "%Y-%m-%d").date()
                eval_date = pred_date + timedelta(days=pred.timeframe_days)
                if today < eval_date:
                    skipped += 1
                    continue

            # Fetch current price
            current_price = get_current_price(pred.ticker_symbol)
            if current_price is None:
                logger.warning(f"Could not fetch price for {pred.ticker_symbol}")
                errors += 1
                continue

            price_at_pred = float(pred.price_at_prediction) if pred.price_at_prediction else None
            if price_at_pred is None or price_at_pred == 0:
                errors += 1
                continue

            # Calculate actual change
            actual_change_pct = ((current_price - price_at_pred) / price_at_pred) * 100

            # Determine actual direction (±2% threshold for neutral)
            if actual_change_pct > 2:
                actual_direction = "up"
            elif actual_change_pct < -2:
                actual_direction = "down"
            else:
                actual_direction = "neutral"

            # Evaluate accuracy
            is_accurate = pred.predicted_direction == actual_direction

            with get_sync_db() as db:
                repository.save_ground_truth(
                    db,
                    prediction_id=pred.id,
                    price_at_evaluation=current_price,
                    actual_change_pct=round(actual_change_pct, 4),
                    actual_direction=actual_direction,
                    is_accurate=is_accurate,
                )
                repository.refresh_channel_accuracy(db, pred.channel_id)

            evaluated += 1
            logger.debug(
                f"Evaluated prediction {pred.id}: {pred.ticker_symbol} "
                f"predicted={pred.predicted_direction} actual={actual_direction} "
                f"accurate={is_accurate} change={actual_change_pct:.2f}%"
            )

        except Exception as e:
            logger.error(f"Error evaluating prediction {pred.id}: {e}")
            errors += 1

    logger.info(
        f"Prediction evaluation complete: "
        f"{evaluated} evaluated, {skipped} skipped (timeframe not elapsed), "
        f"{errors} errors"
    )
    return {
        "evaluated": evaluated,
        "skipped": skipped,
        "errors": errors,
        "total_pending": len(pending),
    }
