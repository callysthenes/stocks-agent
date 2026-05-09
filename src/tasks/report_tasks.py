"""
Celery report tasks — generate and deliver reports via Telegram.
Runs on the default queue.
"""
import asyncio
from datetime import date

from loguru import logger

from src.celery_app import celery_app
from src.models import ReportType
from src.storage import repository
from src.storage.mariadb_client import get_sync_db


@celery_app.task(
    name="src.tasks.report_tasks.send_video_alert",
    bind=True,
    max_retries=2,
    queue="default",
)
def send_video_alert(self, video_id: str) -> dict:
    """
    Generate and send a per-video Telegram alert after ingestion completes.
    """
    logger.info(f"Generating video alert for {video_id}")

    try:
        # Generate report via agent
        loop = asyncio.new_event_loop()
        try:
            from src.agents import run_agent
            with get_sync_db() as db:
                video = repository.get_video_by_id(db, video_id)
                if not video:
                    return {"error": "Video not found"}
                channel_name = video.channel.name if video.channel else "Unknown"
                video_title = video.title
                tickers = list(dict.fromkeys(  # deduplicated, order-preserving
                    m.ticker_symbol for m in video.ticker_mentions
                ))

            query = (
                f"Analiza el vídeo reciente de {channel_name}: '{video_title}'. "
                f"Tickers mencionados: {', '.join(tickers) if tickers else 'ninguno identificado'}. "
                f"Proporciona un análisis breve de los activos mencionados."
            )

            result = loop.run_until_complete(
                run_agent(query=query, send_telegram=True, video_id=video_id,
                          ticker_symbols=tickers)
            )
        finally:
            loop.close()

        final_report = result.get("final_report", "")
        if not final_report:
            return {"error": "Agent produced no report"}

        # Save report to DB
        with get_sync_db() as db:
            report = repository.save_report(
                db,
                report_type=ReportType.video_alert,
                title=f"Alerta: {video_title[:100]}",
                content_markdown=final_report,
                tickers_analyzed=tickers,
                video_id=video_id,
            )

        # Send via Telegram
        msg_id = _send_telegram(final_report)
        if msg_id:
            with get_sync_db() as db:
                repository.mark_report_sent(db, report.id, msg_id)

        return {"video_id": video_id, "report_id": report.id, "telegram_sent": bool(msg_id)}

    except Exception as e:
        logger.error(f"Video alert failed for {video_id}: {e}")
        raise self.retry(exc=e)


@celery_app.task(
    name="src.tasks.report_tasks.send_daily_report",
    bind=True,
    max_retries=2,
    queue="default",
)
def send_daily_report(self) -> dict:
    """
    Generate and send the daily digest report.
    Triggered by Celery Beat at 22:30 UTC.
    """
    logger.info("Generating daily report")
    today = date.today().strftime("%Y-%m-%d")

    try:
        loop = asyncio.new_event_loop()
        try:
            from src.agents import run_agent

            # Build a comprehensive daily query
            with get_sync_db() as db:
                channel_accuracy = repository.get_channel_accuracy_all(db)
                recent_videos = repository.get_recent_videos(db, limit=10)

            channels_info = ", ".join(
                f"{ca.channel.name if ca.channel else 'Unknown'} ({ca.accuracy_pct:.0f}%)" 
                for ca in channel_accuracy
                if ca.accuracy_pct
            ) or "Sin datos aún"

            recent_tickers = set()
            for v in recent_videos:
                for m in v.ticker_mentions:
                    recent_tickers.add(m.ticker_symbol)

            query = (
                f"Genera el resumen diario de hoy {today}. "
                f"Canales monitorizados con precisión: {channels_info}. "
                f"Tickers mencionados recientemente: {', '.join(list(recent_tickers)[:10])}. "
                f"Vídeos procesados hoy: {len(recent_videos)}. "
                f"Incluye análisis de los tickers más mencionados y el estado de las predicciones activas."
            )

            result = loop.run_until_complete(
                run_agent(query=query, send_telegram=True)
            )
        finally:
            loop.close()

        final_report = result.get("final_report", "")
        if not final_report:
            return {"error": "Agent produced no daily report"}

        with get_sync_db() as db:
            report = repository.save_report(
                db,
                report_type=ReportType.daily_summary,
                title=f"Resumen Diario {today}",
                content_markdown=final_report,
                tickers_analyzed=list(recent_tickers),
            )

        msg_id = _send_telegram(final_report)
        if msg_id:
            with get_sync_db() as db:
                repository.mark_report_sent(db, report.id, msg_id)

        return {"date": today, "report_id": report.id, "telegram_sent": bool(msg_id)}

    except Exception as e:
        logger.error(f"Daily report failed: {e}")
        raise self.retry(exc=e)


def _send_telegram(text: str) -> int | None:
    """Send a message via Telegram. Returns message_id on success."""
    from src.config import settings

    if not settings.telegram_chat_id:
        logger.warning("TELEGRAM_CHAT_ID not set — skipping delivery")
        return None

    loop = asyncio.new_event_loop()
    try:
        from telegram import Bot
        async def _send():
            bot = Bot(token=settings.telegram_bot_token)
            msg = await bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=text,
                parse_mode="Markdown",
            )
            return msg.message_id

        return loop.run_until_complete(_send())
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return None
    finally:
        loop.close()
