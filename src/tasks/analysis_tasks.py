"""
Celery analysis tasks — ground truth evaluation, accuracy calculation.
Runs on the default queue (no GPU required).

Price-lookup strategy
---------------------
When a prediction's timeframe has elapsed we use the *historical* closing price
on the evaluation date (via yfinance) rather than today's live price.  This
gives a fair assessment of the prediction as it would have been seen at the time.
A ±5 business-day window is tried if the exact date has no data (holidays, etc.).
"""
from datetime import date, datetime, timedelta

import yfinance as yf
from loguru import logger
from sqlalchemy import select

from src.celery_app import celery_app
from src.models import GroundTruth, Prediction
from src.storage import repository
from src.storage.mariadb_client import get_sync_db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _price_at_date(ticker: str, target: date) -> float | None:
    """
    Return the adjusted closing price for *ticker* on or just before *target*.
    Searches a ±7-day window to handle weekends / holidays.
    Returns None if no data is found.
    """
    start = target - timedelta(days=7)
    end = target + timedelta(days=1)
    try:
        hist = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if hist.empty:
            return None
        # Pick the last available close on or before target_date
        hist.index = hist.index.date  # type: ignore[assignment]
        rows = hist[hist.index <= target]
        if rows.empty:
            return None
        close = rows["Close"].iloc[-1]
        if hasattr(close, "item"):
            close = close.item()
        return float(close)
    except Exception as exc:
        logger.warning(f"[price_at_date] {ticker} @ {target}: {exc}")
        return None


def _current_price(ticker: str) -> float | None:
    """Fallback to live price when no historical data is available."""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        price = fi.last_price or fi.previous_close
        return float(price) if price else None
    except Exception:
        return None


# ── Task ─────────────────────────────────────────────────────────────────────

@celery_app.task(
    name="src.tasks.analysis_tasks.evaluate_pending_predictions",
    bind=True,
    queue="default",
)
def evaluate_pending_predictions(self) -> dict:
    """
    Evaluate all pending predictions whose timeframe has elapsed.

    For each prediction:
      - Resolve the evaluation date = prediction_date + timeframe_days
      - Fetch the historical closing price on that date (yfinance)
      - Compare with price_at_prediction to get actual_change_pct + direction
      - Determine accuracy (predicted_direction == actual_direction, ±2% neutral band)
      - Persist GroundTruth and refresh channel_accuracy summary

    Triggered daily by Celery Beat (default 22:00 UTC).
    """
    logger.info("Starting daily prediction evaluation")

    with get_sync_db() as db:
        pending = repository.get_pending_predictions(db)

    today = date.today()
    evaluated = 0
    skipped = 0
    errors = 0

    for pred in pending:
        try:
            # ── Determine evaluation date ─────────────────────────────────────
            eval_date: date | None = None
            if pred.prediction_date and pred.timeframe_days:
                pred_dt = datetime.strptime(pred.prediction_date, "%Y-%m-%d").date()
                eval_date = pred_dt + timedelta(days=pred.timeframe_days)
                if today < eval_date:
                    skipped += 1
                    continue  # timeframe not yet elapsed
            elif pred.timeframe_days:
                # No prediction_date recorded — use today as baseline
                eval_date = today
            else:
                # No timeframe info — fall back to live price
                eval_date = today

            # ── Fetch price at evaluation date ────────────────────────────────
            price_at_eval = _price_at_date(pred.ticker_symbol, eval_date)
            if price_at_eval is None:
                price_at_eval = _current_price(pred.ticker_symbol)
            if price_at_eval is None:
                logger.warning(f"[eval] No price data for {pred.ticker_symbol}")
                errors += 1
                continue

            price_at_pred = float(pred.price_at_prediction) if pred.price_at_prediction else None
            if price_at_pred is None or price_at_pred == 0:
                logger.warning(f"[eval] No baseline price for prediction {pred.id}")
                errors += 1
                continue

            # ── Compute actual change & direction ─────────────────────────────
            actual_change_pct = (price_at_eval - price_at_pred) / price_at_pred * 100

            if actual_change_pct > 2:
                actual_direction = "up"
            elif actual_change_pct < -2:
                actual_direction = "down"
            else:
                actual_direction = "neutral"

            # ── Also check target-price achievement ───────────────────────────
            target_hit: bool | None = None
            if pred.target_price:
                tp = float(pred.target_price)
                if pred.predicted_direction in ("up", None) and tp > price_at_pred:
                    target_hit = price_at_eval >= tp
                elif pred.predicted_direction == "down" and tp < price_at_pred:
                    target_hit = price_at_eval <= tp

            is_accurate = pred.predicted_direction == actual_direction

            notes = f"eval_date={eval_date.isoformat()}; price_at_eval={price_at_eval:.4f}"
            if target_hit is not None:
                notes += f"; target_price={pred.target_price}; target_hit={target_hit}"

            with get_sync_db() as db:
                repository.save_ground_truth(
                    db,
                    prediction_id=pred.id,
                    price_at_evaluation=round(price_at_eval, 4),
                    actual_change_pct=round(actual_change_pct, 4),
                    actual_direction=actual_direction,
                    is_accurate=is_accurate,
                    notes=notes,
                )
                repository.refresh_channel_accuracy(db, pred.channel_id)

            evaluated += 1
            logger.debug(
                f"[eval] {pred.ticker_symbol} pred={pred.predicted_direction} "
                f"actual={actual_direction} chg={actual_change_pct:.2f}% "
                f"accurate={is_accurate}"
            )

        except Exception as exc:
            logger.error(f"[eval] Error on prediction {pred.id}: {exc}", exc_info=True)
            errors += 1

    logger.info(
        f"Prediction evaluation complete: {evaluated} evaluated, "
        f"{skipped} skipped, {errors} errors (total pending={len(pending)})"
    )
    return {"evaluated": evaluated, "skipped": skipped, "errors": errors}
