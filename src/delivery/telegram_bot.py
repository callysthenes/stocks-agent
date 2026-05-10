"""
Telegram bot — handles commands and delivers reports.
Runs as a standalone process (long-polling).
"""
import asyncio
import sys

from loguru import logger
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.config import settings


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — register the chat ID for report delivery."""
    chat_id = update.effective_chat.id
    logger.info(f"New /start from chat_id={chat_id}")

    # Persist chat_id to settings (runtime only; set TELEGRAM_CHAT_ID in .env for permanence)
    settings.__dict__["telegram_chat_id"] = str(chat_id)

    await update.message.reply_text(
        "¡Hola! Soy StocksAgent, tu analista financiero de IA.\n\n"
        f"Tu chat ID es: `{chat_id}`\n"
        "Añade `TELEGRAM_CHAT_ID=" + str(chat_id) + "` a tu `.env` para recibir informes diarios.\n\n"
        "Comandos disponibles:\n"
        "/analyze TICKER — Análisis completo de una acción\n"
        "/accuracy — Precisión de predicciones por canal\n"
        "/report — Último informe generado\n"
        "/status — Estado del sistema\n"
        "O escribe cualquier pregunta sobre acciones.",
        parse_mode="Markdown",
    )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /analyze TICKER — trigger a full analysis."""
    if not context.args:
        await update.message.reply_text("Uso: /analyze TICKER (ej: /analyze AAPL o /analyze SAN.MC)")
        return

    ticker = context.args[0].upper()
    await update.message.reply_text(f"Analizando {ticker}... esto puede tardar un momento. ⏳")

    try:
        from src.agents import run_agent
        result = await run_agent(
            query=f"Realiza un análisis completo de {ticker}: análisis técnico, fundamental, noticias recientes, y qué han dicho los canales de YouTube sobre esta acción.",
            send_telegram=False,
        )
        report = result.get("final_report", "No se pudo generar el análisis.")
        await update.message.reply_text(report[:4096], parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Analysis error for {ticker}: {e}")
        await update.message.reply_text(f"Error al analizar {ticker}: {e}")


async def accuracy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /accuracy — show channel prediction accuracy."""
    from src.agents.tools.prediction_tools import get_channel_accuracy_summary
    try:
        accuracy_data = get_channel_accuracy_summary()
        if not accuracy_data:
            await update.message.reply_text("Aún no hay datos de precisión disponibles.")
            return

        lines = ["📊 *Precisión de Predicciones por Canal*\n"]
        for ch in accuracy_data:
            acc = ch.get("accuracy_pct")
            total = ch.get("total_predictions", 0)
            accurate = ch.get("accurate_predictions", 0)
            if acc is not None:
                emoji = "🟢" if acc >= 60 else "🟡" if acc >= 40 else "🔴"
                lines.append(
                    f"{emoji} *{ch['channel_name']}*: {acc:.1f}% ({accurate}/{total})"
                )
            else:
                lines.append(f"⚪ *{ch['channel_name']}*: Sin datos suficientes")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error obteniendo datos: {e}")


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /report — retrieve the latest generated report."""
    from sqlalchemy import select
    from src.models import AgentReport
    from src.storage.mariadb_client import get_sync_db

    try:
        with get_sync_db() as db:
            report = db.scalar(
                select(AgentReport)
                .order_by(AgentReport.created_at.desc())
                .limit(1)
            )
        if not report:
            await update.message.reply_text("No hay informes generados aún.")
            return

        text = report.content_markdown[:4096]
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error recuperando informe: {e}")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status — system status check."""
    from src.storage.chroma_client import check_chroma_connection, get_collection_stats
    from src.storage.mariadb_client import check_db_connection
    from src.storage.mariadb_client import get_sync_db
    from src.models import Video, ProcessingStatus
    from sqlalchemy import select, func

    db_ok = check_db_connection()
    chroma_ok = check_chroma_connection()

    try:
        chroma_stats = get_collection_stats() if chroma_ok else {}
        with get_sync_db() as db:
            from sqlalchemy import func
            total_videos = db.scalar(select(func.count(Video.id))) or 0
            completed = db.scalar(
                select(func.count(Video.id))
                .where(Video.processing_status == ProcessingStatus.completed)
            ) or 0
            pending = db.scalar(
                select(func.count(Video.id))
                .where(Video.processing_status == ProcessingStatus.pending)
            ) or 0

        status_msg = (
            "🔧 *Estado del Sistema*\n\n"
            f"{'✅' if db_ok else '❌'} MariaDB\n"
            f"{'✅' if chroma_ok else '❌'} ChromaDB ({chroma_stats.get('count', 0)} chunks)\n\n"
            f"📹 Vídeos totales: {total_videos}\n"
            f"✅ Procesados: {completed}\n"
            f"⏳ Pendientes: {pending}\n"
            f"📊 Dashboard: {settings.dashboard_url}"
        )
        await update.message.reply_text(status_msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error de estado: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle free-text messages — route through the agent."""
    query = update.message.text.strip()
    if len(query) < 3:
        return

    # ── Short-circuit trivial / greeting messages ────────────────────────────
    _GREETINGS = {"hola", "hello", "hi", "hey", "buenas", "buenos días",
                  "buenos dias", "buenas tardes", "buenas noches", "ola", "oi"}
    if query.lower() in _GREETINGS or len(query) <= 10 and not any(
        c.isupper() or c in ".-/" for c in query
    ):
        await update.message.reply_text(
            "¡Hola! Soy StocksAgent 🤖\n\n"
            "Puedo ayudarte con:\n"
            "• `/analyze SAN.MC` — Análisis completo de un ticker\n"
            "• `/accuracy` — Precisión de predicciones por canal\n"
            "• `/report` — Último informe generado\n"
            "• `/status` — Estado del sistema\n\n"
            "O escríbeme directamente tu pregunta sobre una acción, por ejemplo:\n"
            "_¿Qué dicen los canales sobre Santander?_\n"
            "_Analiza técnicamente NVDA_",
            parse_mode="Markdown",
        )
        return

    await update.message.reply_text("Procesando tu consulta... ⏳")

    try:
        from src.agents import run_agent
        result = await run_agent(query=query, send_telegram=False)
        report = result.get("final_report", "No pude generar una respuesta.")
        await update.message.reply_text(report[:4096], parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Agent error for message '{query[:50]}': {e}")
        await update.message.reply_text(
            "Lo siento, ocurrió un error procesando tu consulta. Inténtalo de nuevo."
        )


def run_bot() -> None:
    """Start the Telegram bot (long-polling)."""
    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not set — cannot start bot")
        sys.exit(1)

    logger.info("Starting Telegram bot (long-polling)")
    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("accuracy", accuracy_command))
    app.add_handler(CommandHandler("report", report_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run_bot()
