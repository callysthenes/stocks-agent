"""
Celery analysis tasks — ground truth evaluation, accuracy calculation, Kimball backfill.
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
from sqlalchemy import select, update

from src.celery_app import celery_app
from src.models import GroundTruth, Prediction, Transcript
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


# ── Kimball backfill ──────────────────────────────────────────────────────────

@celery_app.task(
    name="src.tasks.analysis_tasks.backfill_kimball_fields",
    bind=True,
    queue="default",
    time_limit=7200,  # 2 h max (many LLM calls)
)
def backfill_kimball_fields(self) -> dict:
    """
    One-off task: for each video that has predictions missing Kimball fields
    (recommendation, confidence_score, analyst_reasoning), re-run the LLM
    extraction and patch the existing prediction rows in-place.

    Matching strategy: video_id × ticker_symbol (first incomplete match wins).
    Predictions already having all three fields are skipped.
    """
    from src.ingestion.metadata_extractor import MetadataExtractor

    logger.info("Starting Kimball fields backfill")
    extractor = MetadataExtractor()

    # ── 1. Find video_ids that have incomplete predictions ────────────────────
    with get_sync_db() as db:
        rows = db.execute(
            select(Prediction.video_id)
            .where(
                Prediction.recommendation.is_(None),
                Prediction.analyst_reasoning.is_(None),
                Prediction.confidence_score.is_(None),
            )
            .distinct()
        ).scalars().all()
    video_ids = list(rows)
    logger.info(f"Found {len(video_ids)} videos with incomplete predictions")

    updated_total = 0
    skipped_total = 0
    error_total = 0

    for video_id in video_ids:
        try:
            # ── 2. Fetch transcript ───────────────────────────────────────────
            with get_sync_db() as db:
                transcript = db.scalar(
                    select(Transcript).where(Transcript.video_id == video_id)
                )
                if not transcript or not (transcript.cleaned_text or transcript.raw_text):
                    logger.warning(f"[backfill] No transcript for video {video_id}")
                    skipped_total += 1
                    continue
                text = transcript.cleaned_text or transcript.raw_text

            # ── 3. Re-run extraction ──────────────────────────────────────────
            ticker_dicts = extractor.extract_tickers(text)
            if not ticker_dicts:
                logger.info(f"[backfill] No tickers extracted for video {video_id}")
                skipped_total += 1
                continue

            pred_dicts = extractor.extract_predictions(text, ticker_dicts)
            if not pred_dicts:
                logger.info(f"[backfill] No predictions extracted for video {video_id}")
                skipped_total += 1
                continue

            # ── 4. Patch matching predictions ─────────────────────────────────
            # Index extracted predictions by ticker_symbol (last one wins if dup)
            extracted_by_ticker: dict[str, dict] = {}
            for pd in pred_dicts:
                sym = pd.get("ticker_symbol", "").upper().strip()
                if sym:
                    extracted_by_ticker[sym] = pd

            with get_sync_db() as db:
                # Fetch existing incomplete predictions for this video
                incomplete = db.scalars(
                    select(Prediction).where(
                        Prediction.video_id == video_id,
                        Prediction.recommendation.is_(None),
                        Prediction.analyst_reasoning.is_(None),
                        Prediction.confidence_score.is_(None),
                    )
                ).all()

                video_updated = 0
                for pred in incomplete:
                    extracted = extracted_by_ticker.get(pred.ticker_symbol.upper())
                    if not extracted:
                        continue

                    rec = extracted.get("recommendation")
                    if rec not in ("buy", "accumulate", "hold", "reduce", "sell", "avoid"):
                        rec = None

                    db.execute(
                        update(Prediction)
                        .where(Prediction.id == pred.id)
                        .values(
                            recommendation=rec,
                            confidence_score=extracted.get("confidence_score"),
                            analyst_reasoning=extracted.get("analyst_reasoning"),
                            is_long_term=extracted.get("is_long_term"),
                            entry_price=extracted.get("entry_price") or pred.entry_price,
                            stop_loss=extracted.get("stop_loss") or pred.stop_loss,
                            target_price=extracted.get("target_price") or pred.target_price,
                        )
                    )
                    video_updated += 1

                updated_total += video_updated
                logger.info(
                    f"[backfill] video {video_id}: patched {video_updated}/{len(incomplete)} predictions"
                )

        except Exception as exc:
            logger.error(f"[backfill] Error on video {video_id}: {exc}", exc_info=True)
            error_total += 1

    logger.info(
        f"Kimball backfill complete: {updated_total} predictions updated, "
        f"{skipped_total} videos skipped, {error_total} errors"
    )
    return {"updated": updated_total, "skipped": skipped_total, "errors": error_total}
